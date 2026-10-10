# BASIS-A5 — Orthogonal Unique–Shared Local Steering Atlas (SCIENTIFIC SPEC)

Status: **FROZEN PROTOCOL — pre-run additive frozen-steering atlas.** This document
does not reopen DG-00–DG-08, does not change any locked D-test artifact, and does not
introduce a learned controller. It is an **additive, local-only, frozen-direction** stage
that extends the accepted **BASIS-A4** state (repaired Qwen Oracle-local atlas, HEAD
`d00c54c`, final manifest `status=COMPLETE`, 1056 cells) with a new *construction*
(orthogonal Unique/Shared subspace directions) and a new *local* intervention family
(S1/S2/S3). It reuses A4's sites, masks, panels, alignments, NormPreserve, metrics, and
baseline semantics **verbatim**. **No GPU jobs are launched by this stage.**

Base state: BASIS-A4 accepted at `d00c54c` (`BASIS_A4_SPEC.md`,
`BASIS_A4_EXECUTION_PLAN.md`, `results/basis_a4/FINAL_MANIFEST.json`,
`manifests/qwen_local_repair_completeness.json`, `acceptance/qwen_acceptance.json`).

---

## 0. Scope in one paragraph

For **every** encoder and decoder layer of **Whisper-large-v3** and **Qwen3-ASR-1.7B**,
construct — on **CS-Dialogue D-construct only** — two orthogonal unit directions
`v_unique` and `v_shared` from an **uncentered** subspace decomposition of embedded-English
(group A) vs matrix-Mandarin (group B) hidden states at that layer's steering site. Then
evaluate **three Oracle-local** additive interventions (S1 `-v_shared`, S2 `+v_unique`,
S3 `normalize(v_unique - v_shared)`) at `rho = 0.5`, NormPreserve, on the three frozen A4
panels (CS-Dialogue D-dev-select 300; SEAME dev-man 50; SEAME dev-sge 50). **A5 is
LOCAL-ONLY: there are no Global runs, no dose sweep, no Conditioning-Avg, and no new
panels.** A4 Raw-local and Conditioning-local results are **comparators only** and are
never re-decoded.

---

## 1. Prerequisite and exact A4 reuse (FROZEN INVARIANTS)

**A4 prerequisite = PASS** (independently re-verified at HEAD `d00c54c`): the repaired
Qwen Oracle-local masks edit the intended embedded-English positions —
`qwen_local_repair_completeness.json` `status=PASS`, CS
`alignable_nonzero_encoder_rate = 1.0`, `alignable_nonzero_decoder_rate = 1.0`; acceptance
checks `cs_local_nonzero_encoder / cs_local_nonzero_decoder / cs_local_semantic_equality /
cs_local_prefix_exclusion / cs_local_decoder_subset / cs_local_encoder_subset` all `true`;
the 240 old no-op local cells are quarantined under
`quarantine/superseded_broken_qwen_local_mask/`. **If any of these regress, A5 is
BLOCKED.**

A5 inherits the following A4 artifacts **without modification** (a per-cell hash/fingerprint
match is required exactly as in A4 §4, or the cell is not accepted):

| Invariant | Frozen value (A4) |
|---|---|
| A4 protocol hash | `sha256:eb8920e0…deff6` |
| Whisper model revision | `sha256:a8e94b85…fd95` |
| Qwen model revision | `sha256:5d87c842…3453e8` |
| Whisper decoder site | `decoder_post_cross_attn_residual` (DG-02 FROZEN) |
| Whisper encoder site | `encoder_post_self_attn_residual_pre_ffn` |
| Qwen decoder site | `qwen_text_decoder_post_self_attn_residual_pre_mlp` |
| Qwen encoder site | `qwen_audio_encoder_post_self_attn_residual_pre_ffn` |
| Oracle-local masks | reuse A4 accepted masks (§9); Qwen CS uses `cs_ddev_select_alignment_v1` |
| Qwen measured audio fps (median) | `13.007633587786259` (empirically measured, frozen) |
| CS-Dialogue panel | n=300, role `D-dev-select`, fp `sha256:7fb612ee…3398ff` |
| SEAME dev-man panel | n=50, fp `sha256:3f49d08d…e2a3c` |
| SEAME dev-sge panel | n=50, fp `sha256:ebdcdd10…b77238` |
| Repair / NormPreserve | `norm_preserve=true`, `depth_rescale=false`, `rho=0.5` |
| Decoding | greedy, `do_sample=false`, `temperature=0`, `num_beams=1`, `condition_on_prev_tokens=false`, `max_new_tokens=200`; Whisper `language=zh`; Qwen `force_language="Chinese"`, `context=""` |
| Metrics | canonical A4 `metrics_v1` / `result_v1` (§14) |
| Baselines | Whisper baselines reused from A3; Qwen baselines `results/basis_a4/qwen3_asr_1p7b/baseline/` |

Directions and steering are always applied **in the exact space they are constructed in**.
Never mix Whisper (d=1280) and Qwen (dec d=2048 / enc d=1024) hidden spaces.

---

## 2. Extraction population (CS-Dialogue D-construct ONLY)

Vector construction uses **CS-Dialogue D-construct** only; **SEAME never constructs any
vector**. Reuse the **same eligibility population as accepted A4 Raw direction
construction** (baseline-correct spans/positions) wherever possible. The A5 groups are:

- **ENCODER** — at `…encoder…site`, per model:
  - **A (unique/embedded):** baseline-correct **embedded-English acoustic frames**.
  - **B (shared/matrix):** baseline-correct **matrix-language (Mandarin) acoustic frames**.
- **DECODER** — at `…decoder…site`, per model:
  - **A (unique/embedded):** baseline-correct **embedded-English transcript positions**.
  - **B (shared/matrix):** baseline-correct **matrix-language transcript positions**.

Positions/frames are pooled over all D-construct utterances (Qwen construction population:
`n_utterances = 125`, `n_dialogues = 20`; A4 content_counts: embedded_tokens 447, matrix_tokens
362 — both ≥ 64, so the rank grid of §6 is feasible; Whisper counts measured at construction).
Use the **same layer representation/site that will later be steered** — one independent
`(v_unique, v_shared)` pair per (model, side, layer).

---

## 3. Mathematical extraction (uncentered second-moment formulation)

For each (model, side, layer), let the per-position/frame hidden vectors form
`H_A ∈ R^{d×n_A}` (group A) and `H_B ∈ R^{d×n_B}` (group B). The formulation is
**uncentered** — hidden vectors are **not** mean-subtracted before the decomposition (the
group means are computed separately, for sign only, §4).

**Step 1 — truncated bases.** Take the top-`r` left singular vectors of `H_A` and `H_B`:

```
U_A = top-r left singular vectors of H_A          (U_A ∈ R^{d×r},  U_A^T U_A = I_r)
U_B = top-r left singular vectors of H_B
```

Equivalently (and this is the implementation of record, §12), the left singular vectors of
`H` are the eigenvectors of the **uncentered second moment**:

```
C_A = H_A H_A^T / n_A ,   C_B = H_B H_B^T / n_B      (d×d, PSD)
U_A = top-r eigenvectors of C_A ,   U_B = top-r eigenvectors of C_B
```

**Step 2 — principal angles between subspaces.** SVD the `r×r` cross matrix:

```
U_A^T U_B = P Σ Q^T ,     Σ = diag(σ_1 … σ_r),  σ_i ∈ [0,1]  (cosines of principal angles)
a_i = U_A p_i ,   b_i = U_B q_i                    (principal vectors; a_i^T b_i = σ_i)
```

`{a_i}` are orthonormal (in span U_A), `{b_i}` orthonormal (in span U_B), and
`a_i^T b_j = σ_i δ_ij`.

**Step 3 — unique direction (high A-energy, low B-overlap).**

```
E_A(i) = a_i^T C_A a_i
s_unique(i) = E_A(i) · (1 − σ_i^2)
i_u = argmax_i  s_unique(i)
v_unique = a_{i_u}
```

**Step 4 — shared direction (high overlap, energy in both).**

```
E_B(i) = b_i^T C_B b_i
s_shared(i) = σ_i^2 · sqrt( E_A(i) · E_B(i) )
i_s = argmax_{i ≠ i_u}  s_shared(i)
b'_{i_s} = sign(a_{i_s}^T b_{i_s}) · b_{i_s}          # align paired directions before averaging
v_shared = normalize( a_{i_s} + b'_{i_s} )
```

**Step 5 — required constraints.**

```
require  i_s ≠ i_u
verify   | v_unique^T v_shared | ≤ 1e-3   (orthogonality tolerance)
```

**Why orthogonality is exact by construction (documented rationale, not a fitted claim):**
`v_unique = a_{i_u}` and `v_shared ∝ a_{i_s} + b'_{i_s}`. Since `a_{i_u} ⊥ a_{i_s}`
(distinct principal vectors of the same orthonormal set) and
`a_{i_u}^T b_{i_s} = σ_{i_u} δ_{i_u,i_s} = 0` when `i_u ≠ i_s`, the two directions are
orthogonal in exact arithmetic. The `≤ 1e-3` check catches numerical drift only; a layer
that fails it is **flagged** (§13), not silently patched.

---

## 4. Sign orientation — REQUIRED (frozen before any recognition outcome)

Principal-vector signs are arbitrary; the **steering sign is frozen from construction
geometry only**, never from recognition performance.

```
mu_A = mean_column(H_A) ,   mu_B = mean_column(H_B)      # uncentered means, sign use only
delta_AB   = normalize(mu_A − mu_B)                       # the A4 Raw-style difference-in-means axis
pooled_anchor = mu_A + mu_B
```

- **Unique:** if `dot(v_unique, delta_AB) < 0` then `v_unique ← −v_unique`.
  ⇒ `+Unique` always points toward the A-vs-B (embedded-vs-matrix) contrast.
- **Shared:** if `dot(v_shared, pooled_anchor) < 0` then `v_shared ← −v_shared`.
  - **Degeneracy fallback (documented):** if `‖pooled_anchor‖ < 1e-8·max(‖mu_A‖,‖mu_B‖)`
    (numerically degenerate), fall back to orienting `v_shared` toward `mu_A`
    (`dot(v_shared, normalize(mu_A))`); if that is also degenerate, keep the raw sign and
    record `sign_fallback = "indeterminate"` in diagnostics. The fallback never consults
    steering outcomes.

Signs are sealed in the frozen direction artifact before any decode.

---

## 5. Final unit directions

All final directions are **unit-norm** and applied **Oracle-local** at `rho=0.5`,
NormPreserve.

| ID | Name | Direction |
|---|---|---|
| **S1** | Minus-Shared | `d_neg_shared = −v_shared` |
| **S2** | Add-Unique | `d_pos_unique = +v_unique` |
| **S3** | **Unique-minus-Shared composite (U-S composite)** | `d_unique_minus_shared = normalize(v_unique − v_shared)` |

Because the pair is orthonormal, `‖v_unique − v_shared‖ ≈ sqrt(2)`, so S3 equals
`(v_unique − v_shared)/sqrt(2)` within tolerance (the realized norm is recorded).

**Terminology (mandatory):** S3 is **NOT** "difference-in-means." The existing A4 **Raw**
vector is the difference-in-means direction. S3 is the **Unique-minus-Shared composite**
(U-S composite). Any report/table/figure using "difference-in-means" for S3 is a naming
defect.

---

## 6. Rank / stability gate (construction data only)

- **Primary steering rank: `r = 32`** for both groups, both models, both sides.
- **Observation-count guard:** `r_eff = min(32, n_A, n_B)`; if `min(n_A, n_B) < 32` the
  (model, side, layer) is **flagged** and `r_eff` reduced accordingly (the selection and
  orthogonality logic are unchanged). Layers with `n_A < 8` or `n_B < 8` are marked
  `insufficient_support` and excluded from claims (still recorded).
- **Stability grid (pre-GPU, construction data only):** at the A4 anchor/candidate layers,
  recompute `(v_unique, v_shared)` at `r ∈ {16, 32, 64}` (each capped at
  `min(n_A, n_B)`) and report **cosine stability** `|cos(v_r, v_32)|` for both directions.
- **Selection rule:** rank is **fixed at 32 a priori**. Dev-panel MER/PIER are **NEVER**
  used to select rank. The grid is a **gate**, not a tuner: if `r = 32` is *clearly
  unstable* (e.g. median `|cos(v_16,v_32)|` or `|cos(v_64,v_32)|` < 0.90 across anchor
  layers for a direction), **BLOCK** the GPU atlas and escalate as an `OPEN DECISION` in
  `METHOD_CONTRACT.md` — do not silently pick a downstream-better rank.

---

## 7. Models / depth

- **Whisper-large-v3:** encoder **E0–E31** (32), decoder **D0–D31** (32), d_model=1280.
- **Qwen3-ASR-1.7B:** audio-encoder **E0–E23** (24, d=1024), text-decoder **D0–D27**
  (28, d=2048) — the exact frozen A4 layer counts.

All three A5 directions (S1, S2, S3) run at **every** encoder and decoder layer of both
models.

---

## 8. Datasets (exact frozen A4 panels)

- CS-Dialogue **D-dev-select = 300** (role D-dev-select).
- SEAME **dev-man = 50**, SEAME **dev-sge = 50** (evaluation only; never construction).
- No resampling, no expansion. **`D-dev-confirm` DO NOT USE. `D-test` DO NOT USE.**

---

## 9. Steering scope — LOCAL ONLY

A5 has **no Global runs**.

- **Encoder Local:** edit only reference-aligned **embedded-English acoustic frames**.
- **Decoder Local:** edit only reference-aligned **embedded-English transcript/generation
  positions**.
- **Reuse the exact accepted A4 Oracle-local masks** (Whisper A3-derived spans mapped to
  each model's frame grid; Qwen CS `cs_ddev_select_alignment_v1`, SEAME frozen target
  segments). **Do NOT invent a new localization implementation.** Never edit
  prompt/language-control, audio-prefix/placeholder, padding, BOS/EOS/special positions;
  the local mask is a strict subset of the global-eligible transcript/frame set (enforced
  by the A4 acceptance checks already frozen).

---

## 10. Intervention

- Dose **`rho = 0.5`** for the full-depth map (single dose; **no** dose sweep). Dose
  characterization on promising layers is explicitly **future work**.
- **NormPreserve** (existing kernel), **no depth rescale**, `g=0`/`rho=0` identity holds
  (inherited, frozen).
- Cross-model fairness logging is mandatory per cell (as A4 §9): original hidden norm, mean
  perturbation norm, total intervention energy, relative perturbation, edited count, edited
  fraction. `rho=0.5` is **not** treated as physically dose-matched across architectures.

---

## 11. Experiment matrix and expected counts

| Model | Layers (enc+dec) | Directions | Datasets | Cells |
|---|---|---|---|---|
| Whisper | 64 (32+32) | 3 (S1,S2,S3) | 3 | **576** |
| Qwen | 52 (24+28) | 3 | 3 | **468** |
| **Total** | | | | **1044** |

Breakdown by side: encoder = Whisper 288 + Qwen 216 = **504**; decoder = Whisper 288 +
Qwen 252 = **540**. **All 1044 are new decode work** (A5 directions are new). Scope is
`oracle_local` for every cell (`rho=0.5`). Conditioning-Avg is not part of A5. Duplicate
scientific keys must be 0; empty provenance must be 0; every cell file must exist with exact
layer coverage.

**Comparators (NOT re-decoded):** frozen A4 `baseline`, A4 **Raw Oracle-local**, and A4
**Conditioning Oracle-local** (decoder only) at matching (model, dataset, layer).

---

## 12. Construction efficiency (streaming, no full activation cache)

Because the left singular vectors of `H` are the eigenvectors of `H Hᵀ`, A5 construction
**streams** per (model, side, layer, group):

- running `count` (`n`), running `sum` (for `mu`), running **uncentered second moment**
  `S = Σ h hᵀ` (d×d), accumulated in **float64** for numerical stability.

Then `C = S / n`; eigendecompose `C` (d ≤ 2048 ⇒ trivial) to obtain `U_A`, `U_B`; proceed
with §3–§4. This avoids storing all hidden vectors. Capture **all layers in shared
model passes** (one forward per construction utterance records every layer/side/group). Peak
second-moment memory ≈ (#layers×2 groups × d² × 8 B): ~1.7 GB Whisper, ~4 GB Qwen —
tractable; no giant activation cache. Direction `.npy` + a per-layer diagnostics JSON are
persisted; raw activations are not.

---

## 13. Required direction diagnostics (per model/side/layer)

Save, for every (model, side, layer): `n_A`, `n_B`, `r_eff`, `i_unique`, `i_shared`,
`sigma_unique` (`σ_{i_u}`), `sigma_shared` (`σ_{i_s}`), `E_A(unique)`, `E_A(shared)`,
`E_B(shared)`, `s_unique`, `s_shared`, `norm_unique`, `norm_shared`,
`cos(unique, shared)`, `cos(unique, Raw_A4)`, `cos(shared, Raw_A4)`,
`cos(unique, Conditioning_A4)` and `cos(shared, Conditioning_A4)` (**decoder layers only**;
Conditioning is decoder-only), **projection energy of A and B onto Unique**
(`‖v_uᵀ H_A‖²/n_A`, `‖v_uᵀ H_B‖²/n_B`), **projection energy of A and B onto Shared**,
`sign_flip_unique`, `sign_flip_shared`, `sign_fallback`, and artifact hashes
(`direction_hash`, `site_hash`, `mask_hash`, `construction_config_hash`,
`panel_fingerprint`, `model_revision`). The orthogonality tolerance (`|cos(unique,shared)|
≤ 1e-3`) must **pass** (or the layer is flagged and excluded from claims). The Raw/Cond
comparison vectors are the frozen A4 directions at the same (model, side, layer); this
geometry is **within-model only** — never compare coordinates across models.

---

## 14. Metrics (reuse A4 canonical)

`MER`, `PIER`, `EN-WER`, matrix `CER`, `poi_corrections`, `poi_corruptions`,
`poi_net_utility = poi_corrections − poi_corruptions`, `outside_harm` where valid,
`embedded_retention`, `matrix_retention`, insertions/deletions/substitutions,
`edited_positions_or_frames`, `total_intervention_energy` (+ the §10 fairness logs). No new
metric implementation. `poi_net_utility` identity must hold for every accepted row.

---

## 15. Comparators

Final analysis compares A5 (S1/S2/S3) against the **frozen** A4 `baseline`, `Raw
Oracle-local`, and `Conditioning Oracle-local` at matching (model, dataset, layer, side).
These comparators are read from `results/basis_a4/` and are **never re-run**.

---

## 16. Scientific questions

Q1. Does **+Unique (S2)** outperform **Raw-local**? Q2. Does **−Shared (S1)** help, or does
it mostly remove useful common structure? Q3. Does **U-S composite (S3)** yield a better
correction/damage tradeoff? Q4. Is **Unique geometrically distinct from Raw**
(`cos(unique, Raw)` well below 1)? Q5. Does **Shared** carry comparable energy/support in A
and B (`E_A(shared) ≈ E_B(shared)`, high `σ_shared`)? Q6. Are effects primarily
**decoder-side**, or can **encoder Unique** steering help? Q7. Do useful **depth regions
reproduce Whisper → Qwen** (normalized relative depth `layer/(L−1)`)? Q8. Do **signs/trends
transfer CS → SEAME** (exploratory, n=50)? Q9. Does the decomposition produce **useful
mechanistic separation**, or merely reparameterize the existing Raw contrast? Answers are
not forced positive.

---

## 17. Scientific risks

1. **Reparameterization risk (Q9 is the headline null).** If `S3 ≈ Raw-local` behaviorally
   and `cos(unique, Raw)` is high, A5 adds no mechanism. Mitigation: report geometry
   (§13) and the S3-vs-Raw behavioral delta explicitly; a null result is a valid,
   reportable outcome — do not spin it.
2. **Uncentered decomposition is dominated by the mean / DC component.** Since no centering
   is applied, top eigenvectors may track overall magnitude rather than contrastive
   structure. Mitigation: this is the **deliberate** user-proposed formulation; document it,
   report projection energies, and keep the mean explicit in sign logic only.
3. **Rank instability at r=32.** Mitigation: §6 pre-GPU stability gate on construction data;
   BLOCK if unstable; never tune rank from dev outcomes.
4. **Sign leakage.** Choosing signs from steering performance would invalidate causal
   claims. Mitigation: §4 freezes signs from construction geometry before any decode.
5. **Small support for group A (embedded English) at some layers/decoder.** Mitigation:
   observation-count guard (§6); `insufficient_support` flag; excluded from claims.
6. **Oracle-local is an upper bound, not deployable.** Mitigation: state it in every A5
   claim; A5 tests mechanism, not a deployable localizer.
7. **Cross-model / cross-corpus over-reading.** Mitigation: normalized relative depth for
   Whisper↔Qwen pattern comparison only; no hidden-space equivalence; no `rho=0.5`
   energy-matching; SEAME (n=50) is exploratory, never confirmatory.
8. **Comparator drift.** Mitigation: strict per-cell A4 hash/fingerprint/site/mask/panel
   match before any A5 cell is accepted; comparators read-only.

---

## 18. Protocol Status: **PASS**

All extraction semantics, sign conventions, ranks, populations, sites, masks, panels,
counts, diagnostics, metrics, and comparators are determined and frozen. Execution in
`BASIS_A5_EXECUTION_PLAN.md`. No GPU work is launched by this stage.
