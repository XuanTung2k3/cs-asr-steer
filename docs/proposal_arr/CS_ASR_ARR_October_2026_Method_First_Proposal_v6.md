# Selective Test-Time Steering for Code-Switching ASR

## Method-First Research Proposal — v6 (Contrastive Basis + Adaptive Damage-Aware Controller)

**Version:** v6, 2026-09-07 (supersedes v5 for all post-DG-02 work)
**Status:** ACTIVE — scientific source of record for the current implementation contract
(`docs/current/METHOD_CONTRACT.md`).
**Primary model:** Whisper-large-v3, frozen backbone.
**Primary corpus:** CS-Dialogue (Mandarin–English spontaneous dialogue).
**Intervention site:** decoder post-cross-attention residual, pre-FFN — **FROZEN (DG-02)**.
**Short idea:** keep a small **contrastive steering basis fixed** at the frozen exact site, and learn
a **per-token adaptive controller** that decides *how strongly* and *in which basis direction* to
steer, trained to **correct** embedded-language errors while **preserving** everything the baseline
already got right.

> **Provenance.** This v6 proposal is the assembled write-up of the updated method specified in the
> DG-03R reconciliation (2026-09-07). It replaces v5's *where/whether/how-fixed* localizer +
> utility-selector + factorized-gate design. v5
> (`CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`) is **SUPERSEDED / HISTORICAL** and is kept
> only for provenance; where v5 disagrees with this file, **v6 wins** for post-DG-02 work. This
> document adds no scientific claim beyond the supplied specification.

---

## 1. Motivation

Multilingual ASR models such as Whisper transcribe monolingual speech well but systematically
mishandle **intra-sentential code-switching**: short embedded-language words (here, embedded English
inside a Mandarin matrix) are dropped, transliterated into the matrix script, or otherwise
misrecognized, even when the surrounding matrix-language material is correct. Retraining the
backbone is expensive and risks regressions on the (already good) matrix language.

DG-02 froze an exact **test-time intervention site** — the decoder post-cross-attention residual,
before the FFN — where a rank-one edit to the residual stream can be applied per token during
decoding, with norm-preserving repair and no change to the frozen weights. The open scientific
question is **what to add there, and how much, at each token**, such that embedded-language errors
are corrected without damaging correct content. v6 answers this with a **fixed contrastive basis**
plus a **learned adaptive controller**, trained under an explicit **damage-aware** objective.

## 2. Central hypothesis

A small, frozen **contrastive basis** `V^0 = [v_local, v_cond]` constructed at a single decoder layer
(L16 or L24) spans the directions relevant to embedded-vs-matrix language behavior at the exact
site. A lightweight controller that reads only the inference-available site state `LN(r_{ℓ,t})` can
predict, **per token and with no oracle code-switch location**, a strength `g_t` and a basis mixture
`π_t` that repair embedded-language errors while a damage-aware training objective bounds collateral
harm to correct embedded- and matrix-language content.

## 3. Research questions

- **RQ1 — Frozen-steering headroom.** Does any *fixed* steering along the basis (global local, or a
  fixed local+conditioning mixture) produce a favorable point on the free-decoding
  **correction–damage frontier** at L16 or L24? (If no fixed action helps, no learned controller can.)
- **RQ2 — Adaptive controller.** Does the learned per-token controller `f_θ(LN(r_t)) → (g_t, π_t)`
  dominate the best *fixed* operating point — more corrections at equal-or-lower damage — using no
  oracle location at inference?
- **RQ3 — Method comparison.** Is the controller non-dominated versus a SALSA-style learned global
  vector and a matched-data LoRA on the correction–retention–efficiency trade-off?

---

## 4. Contrastive steering basis (per candidate layer ℓ ∈ {16, 24})

Constructed on `D-construct` only, at the frozen exact site, read from the pre-intervention site
state `r_{ℓ,t}`; every vector is finite and unit-norm; artifacts are versioned/hashed and frozen
before any confirmatory run.

**Raw language contrast** (difference of mean site states at baseline-correct positions):
```
v_ℓ^{raw} = μ_{ℓ,E}^{correct} − μ_{ℓ,M}^{correct}
```
where `μ_{ℓ,E}^{correct}` / `μ_{ℓ,M}^{correct}` are the mean `r_{ℓ,t}` over baseline-**correct**
embedded-language / matrix-language positions.

**Language-conditioning direction** (same audio/reference prefix decoded under embedded- vs
matrix-language conditioning `c_E` / `c_M`):
```
v_ℓ^{cond} = normalize( E[ r_{ℓ,t}(c_E) − r_{ℓ,t}(c_M) ] )
```

**Conditioning-residualized local-language direction** (remove the conditioning component):
```
v_ℓ^{local} = normalize( v_ℓ^{raw} − ⟨v_ℓ^{raw}, v_ℓ^{cond}⟩ v_ℓ^{cond} )
```

**Initial (rank-2) basis** (frozen; the controller mixes its columns):
```
V_ℓ^0 = [ v_ℓ^{local}, v_ℓ^{cond} ]
```

**Terminology (conservative).** `v_local` = *conditioning-residualized local-language direction*;
`v_cond` = *language-conditioning direction*. Neither is a "pure acoustic direction", a "universal
language axis", or a "causal language representation". The legacy post-FFN direction `v_nat` is
**not** `v_local` and must not be relabeled as such.

## 5. Adaptive controller (proposed core method)

The controller reads the exact-site state and predicts, per token, both the intervention strength and
a mixture over the frozen basis:
```
(g_t, π_t) = f_θ( LN(r_{ℓ,t}) )
d_{ℓ,t}    = normalize( V_ℓ^0 π_t )
r̃_{ℓ,t}   = NormPreserve( r_{ℓ,t} + β · g_t · d_{ℓ,t} )
```
- `g_t ∈ [0,1]` — per-token intervention strength (gate).
- `π_t` — per-token mixture weights over the columns of `V_ℓ^0`; `d_{ℓ,t}` is rank-one per position.
- `β` — global steering strength (fixed hyperparameter or a single learned scalar).
- `NormPreserve` is the frozen DG-02 repair (per-token L2-norm preservation of the site; no
  `sqrt(num_layers)` rescale). Zero-gain positions are bit-identical; `β=0` is a no-op.
- `f_θ` is a small **bottleneck** module; `LN` is a layer-norm on the site state.
- **No oracle code-switch location at inference:** the only input is `LN(r_{ℓ,t})`, an
  inference-available site state with no reference/alignment/oracle-span dependence. The controller
  runs per row, per token on the frozen DG-02 interface (beam-safe by construction).

**Main method:** basis `V^0` fixed; learn the controller (gate + mixture), optionally learn/tune `β`.

**Constrained basis-refinement variant (ablation, not default):**
```
V_ℓ = V_ℓ^0 + ΔV_ℓ ,   anchor regularizer 𝓛_anchor = ‖ΔV_ℓ‖_F²
```
The anchor keeps the refined basis close to the frozen `V^0`.

**SALSA-style global learned vector (baseline, not the method):** a single unconstrained learned
global steering vector at the layer — a learned-direction comparator, not the proposed controller.

## 6. Damage-aware training

Trains **only** the controller `f_θ` (and optionally `β`); the backbone stays frozen and `V^0` stays
fixed (except in the refinement variant). Position sets, defined against the frozen baseline (no
intervention):
- `𝒞_E` — baseline-**wrong** embedded-language positions (to be corrected);
- `ℛ_E` — baseline-**correct** embedded-language positions (to be preserved);
- `ℛ_M` — baseline-**correct** matrix-language positions (to be preserved).

Objectives:
```
𝓛_corr  = (1/|𝒞_E|) Σ_{t∈𝒞_E}  − log p_θ(y_t | x, y_{<t})
𝓛_ret,E = (1/|ℛ_E|) Σ_{t∈ℛ_E}  D_KL( p_{0,t} ‖ p_{θ,t} )
𝓛_ret,M = (1/|ℛ_M|) Σ_{t∈ℛ_M}  D_KL( p_{0,t} ‖ p_{θ,t} )
𝓛       = 𝓛_corr + λ_E 𝓛_ret,E + λ_M 𝓛_ret,M + λ_A 𝓛_anchor
```
- `p_{0,t}` = frozen-baseline next-token distribution; `p_{θ,t}` = distribution with the controller active.
- Correction is **correction-set-only** cross-entropy on `𝒞_E`.
- Retention is an **explicit KL-to-baseline** loss on `ℛ_E` and `ℛ_M`.
- `𝓛_anchor` is active **only** for the basis-refinement variant; for the main method `λ_A` is off.
- An **optional gate-coverage penalty** may be added *only if* measured controller coverage is
  excessively broad. Do not claim sparsity unless coverage is genuinely sparse.

**Teacher forcing vs free decoding.** Teacher forcing is used for **optimization** (the losses
above). **Free decoding** is used for **checkpoint/system selection and all practical ASR claims.**
A teacher-forced improvement is never itself a transcript claim.

---

## 7. Nested experimental design (Whisper-first core study)

The design is nested: first confirm the basis is causal (DG-03), then bound what *fixed* steering can
do (DG-04), then learn the controller (DG-05) and train it damage-aware (DG-06), then situate it
against learned baselines and ablate its parts (DG-07), then lock the core evaluation (DG-08). Scope
is deliberately **Whisper-large-v3 × CS-Dialogue** until the core method is established.

### DG-03 — Basis + causal validation
Whisper-large-v3 × CS-Dialogue, at L16 and L24: construct `v_raw`, `v_cond`, `v_local`; assemble and
**version/hash** `V^0`; verify finite/unit-norm/dialogue-bootstrap stability; run the intended-direction
**causal screen** and its controls (**sign-reversed**, **matched-norm random**, **wrong-location**);
**select L16 or L24**. Purpose: verify steering headroom and whether the basis is scientifically useful.

### DG-04 — Frozen steering baselines (RQ1)
Frozen backbone; global fixed **local** steering; fixed **local+conditioning** mixture; an exact-site
F5-style/projection-gated baseline where scientifically comparable; a small steering-strength grid.
Purpose: establish the **correction–damage frontier** before any learning.

### DG-05 — Adaptive controller
Implement `f_θ(LN(r_t)) → (g_t, π_t)` with `V^0` fixed; **no oracle CS location** at inference.

### DG-06 — Damage-aware training
Staged: (1) correction-only → (2) + matrix retention → (3) + embedded retention → (4) optional gate
penalty if coverage is broad → (5) optional constrained basis refinement.

### DG-07 — Learned baselines + core ablations
SALSA-style learned global vector; matched-data LoRA; local-only; conditioning-only; fixed vs learned
mixture; global vs adaptive strength; fixed vs refined basis; correction-only vs correction+retention.

### DG-08 — Locked Whisper core evaluation
Whisper-large-v3 × CS-Dialogue: three seeds for trained systems; greedy primary + beam-5 finalists;
efficiency; conversation/dialogue-block bootstrap intervals; the **correction–damage frontier**.
Freeze the entire pipeline **before** `D-test`; run `D-test` once, mechanically.

### Later (only after the Whisper core method succeeds)
SEAME; ViMedCSS; selected multilingual datasets; Qwen3-ASR (architecture replication first); other
speech/audio-language models if feasible. Do not expand architecture/dataset scope before the core
study is established.

## 8. Baselines

Frozen backbone; global fixed local steering; fixed local+conditioning steering; exact-site
F5/projection-gated steering where comparable; **SALSA-style learned global** steering; **matched-data
LoRA**. Optional-only (never blockers): adapter/ReFT; full fine-tuning.

## 9. Ablations

Local-only vs conditioning-only vs `V^0`; fixed mixture vs learned token mixture; global vs adaptive
strength; fixed vs refined basis; correction-only vs +matrix vs +embedded retention (staged).

## 10. Evaluation and efficiency

Primary comparison: the **correction–retention–efficiency trade-off**, visualized as the
**correction–damage frontier** over steering strength / controller operating points. Report per
operating point: MER, PIER, embedded-language WER, matrix-language CER/WER, correction rate,
corruption rate, outside-harm/edit diagnostics (with the frozen DG-01 semantic distinction), embedded
retention, matrix retention, controller gate-strength distribution, and gate coverage *once its
denominator is legitimately defined*. **Efficiency:** trainable parameters, GPU-hours, memory,
latency. Confidence intervals are conversation/dialogue-block bootstrap. Metric definitions and the
gain sign (`baseline − method`, positive = improvement) are **frozen by DG-01** and unchanged here.

## 11. Limited mechanistic controls

Essential causal controls (DG-03): **sign-reversed** direction; **matched-norm random** direction;
**wrong-location**; **preservation / correct-token** analysis. Encoder–decoder disagreement may be
reported only as **optional supporting analysis**, never as a decision gate. No new mechanistic
analyses are introduced.

---

## 12. Intended contributions

1. A method for **selective test-time steering of code-switching ASR** at a frozen exact
   decoder site (post-cross-attention, pre-FFN), keeping the backbone unchanged.
2. A **frozen contrastive basis** (`v_local` + `v_cond`) plus a **learned per-token adaptive
   controller** that sets steering strength and basis mixture from the site state alone, with **no
   oracle code-switch location at inference**.
3. A **damage-aware training** formulation (correction-set CE + explicit KL retention on correct
   embedded and matrix content, optional basis-refinement anchor) and a **correction–damage frontier**
   evaluation of the correction/retention/efficiency trade-off.
4. A Whisper-large-v3 × CS-Dialogue study situating the controller against fixed steering, a
   SALSA-style learned global vector, and matched-data LoRA.

## 13. Claim boundaries

The paper/results must **not** claim: that the system is training-free (the backbone is frozen, but
the controller is trained); that a basis vector is a pure acoustic direction, a universal language
axis, or a causal language representation; that every embedded-language span needs steering; that a
frame-level LID gain is ASR correction; that a teacher-forced gold-token improvement is a transcript
improvement; that successful intervention proves the unmodified model normally uses this direction;
or multilingual/architecture generality from one model and one language pair. Safe framing:
oracle-location steering is an **upper bound**, not a deployable system; the pipeline is frozen
**before** `D-test` and the primary claim is never chosen after seeing test outcomes.

---

## Appendix A — Relationship to v5 (superseded)

v5 framed the system as *learn **where** code-switching occurs (temporal localizer) and **whether**
steering helps (outcome-supervised utility selector + factorized gate, gated partly by
encoder–decoder disagreement), while keeping **how** to steer fixed.* v6 **replaces** that
where/whether machinery with a single learned **adaptive controller** over a frozen contrastive
basis, trained under an explicit damage-aware objective. Consequently the temporal localizer, the
outcome-supervised utility selector, the abstention selector, the factorized gate, and the
disagreement **decision gate** are **not** part of the v6 core method (disagreement survives only as
optional supporting analysis). The DG-01 metric definitions and the DG-02 exact site, `r=q+u_source`,
NormPreserve, forced-prefix exclusion, cache handling, and candidate layers {16, 24} are unchanged and
carried into v6.
