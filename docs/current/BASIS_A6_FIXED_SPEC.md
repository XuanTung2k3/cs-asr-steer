# BASIS-A6-F — Corpus-Level Fixed Steering (SCIENTIFIC SPEC)

Status: **FROZEN PROTOCOL — cross-corpus fixed-direction steering.** Additive, Oracle-local,
frozen-direction stage. It reuses BASIS-A4 sites/masks/metrics/baseline semantics and BASIS-A5
Unique/Shared construction **verbatim**, and extends them along two new axes: **two
construction sources** (`CS-D-construct`, `ASCEND-construct`) and **four evaluation panels**
(CS `D-dev-select`, SEAME `dev-man`, SEAME `dev-sge`, `ASCEND-eval`). It introduces two
decoder-only conditioning variants (**Conditioning-CS**, **Conditioning-All**), a **dose
sweep** `ρ ∈ {0.5,1,2,4,6}`, and **two decoding conditions** (`greedy`, `official_standard`).
**No GPU jobs and no scientific inference are launched by this stage.** Part of the BASIS-A6
freeze (`BASIS_A6_EXPANDED_SPEC.md`).

Base state: BASIS-A5 COMPLETE (`results/basis_a5_unique_shared/FINAL_MANIFEST.json`,
`status=COMPLETE`), BASIS-A4 accepted at `d00c54c`. Companions:
`BASIS_A6_ORACLE_TT_SPEC.md`, `BASIS_A6_ASCEND_DATA_SPEC.md`, `BASIS_A6_EXECUTION_PLAN.md`.

---

## 0. Scope in one paragraph

Construct **one frozen direction per (construction source × model × side × layer × direction
type)** from **many** utterances of a construction subset, then evaluate each frozen direction
**unchanged** on all four panels. This measures (i) within-corpus steering, (ii) cross-corpus
direction **transfer**, and (iii) direction-**source** dependence. There is **no Global**
steering and **no per-utterance** construction in A6-F (per-sample directions are the separate
A6-TT branch).

---

## 1. Models / depth (verify counts from frozen config — FROZEN)

- **Whisper-large-v3:** encoder **E0–E31** (32), decoder **D0–D31** (32), `d_model = 1280`.
- **Qwen3-ASR-1.7B:** audio-encoder **E0–E23** (24, d=1024), text-decoder **D0–D27**
  (28, d=2048).

These are the frozen A4/A5 layer counts (BASIS_A4_SPEC §4–§5, verified from `config.json`).
Directions are always constructed and applied **in the exact hidden space of that model/side**;
never mix Whisper and Qwen spaces.

---

## 2. Construction sources (TWO — independent)

Directions are constructed **independently** for each source:

1. **`CS-D-construct`** — CS-Dialogue `D-construct` (the frozen A4/A5 construction population:
   125 utterances / 20 dialogues; embedded/matrix content counts per A4/A5).
2. **`ASCEND-construct`** — the frozen `ASCEND_CONSTRUCT_MANIFEST.json` subset (ASCEND train
   only, size-matched to CS D-construct; `BASIS_A6_ASCEND_DATA_SPEC.md`).

SEAME **never constructs any vector.** Evaluation panels never construct. Thus each direction
type exists in two source variants, e.g. `v_raw^(CS,l)` and `v_raw^(ASCEND,l)`,
`v_unique^(CS,l)` and `v_unique^(ASCEND,l)`, etc.

**Eligibility for construction** reuses the accepted A4/A5 baseline-correct rule per corpus:
group A = embedded-English positions/frames, group B = matrix-language (Mandarin/Chinese)
positions/frames, pooled over **all** eligible representations of the construction subset. For
CS this is the exact A4/A5 population. For ASCEND it is the canonical language-tag eligibility
(`BASIS_A6_ASCEND_DATA_SPEC §4`) applied at the same sites. Never construct per utterance in
A6-F.

---

## 3. Intervention sites (FROZEN — reuse A4/A5 verbatim)

| Model | Side | Site (frozen) |
|---|---|---|
| Whisper | decoder | `decoder_post_cross_attn_residual` (DG-02 FROZEN, pre-FFN) |
| Whisper | encoder | `encoder_post_self_attn_residual_pre_ffn` |
| Qwen | decoder | `qwen_text_decoder_post_self_attn_residual_pre_mlp` |
| Qwen | encoder | `qwen_audio_encoder_post_self_attn_residual_pre_ffn` |

Exact-site primitive: `csasr.lss.sites.DecoderPostCrossAttnSteeringHook` and the Qwen site
reconstructions (BASIS_A4_SPEC §6). NormPreserve kernel; **no** `sqrt(num_layers)` /depth
rescale; `g=0`/`ρ=0` identity holds.

---

## 4. The six steering methods (FROZEN)

| ID | Name | Direction `d` | Sides |
|---|---|---|---|
| **A6-R** | Raw | `normalize(mu_E − mu_M)` | encoder + decoder |
| **A6-U** | Add-Unique | `+v_unique` | encoder + decoder |
| **A6-S** | Minus-Shared | `−v_shared` | encoder + decoder |
| **A6-US** | Unique-minus-Shared | `normalize(v_unique − v_shared)` | encoder + decoder |
| **A6-C-CS** | Conditioning-CS | `v_cond_cs` (§6) | **decoder only** |
| **A6-C-ALL** | Conditioning-All | `v_cond_all` (§7) | **decoder only** |

- **A6-R** = the frozen A4 **Raw** difference-in-means direction (BASIS_A4_SPEC §3), one per
  (source, model, side, layer). It is the "difference-in-means" axis; A6-US is **not**.
- **A6-U / A6-S / A6-US** = the frozen A5 constructions S2/S1/S3 (BASIS_A5_SPEC §3–§5):
  uncentered second-moment subspace decomposition → principal angles → `v_unique` (high
  A-energy, low B-overlap) and `v_shared` (high overlap, energy in both), signs frozen from
  construction geometry only (A5 §4), orthogonality `|v_unique·v_shared| ≤ 1e-3` verified.
  Primary rank `r = 32` with the A5 observation-count guard `r_eff = min(32, n_A, n_B)`.
- **A6-C-CS / A6-C-ALL** = the paired-condition conditioning directions (§6–§7), decoder only.

**No Global A6 steering.** Every method is Oracle-local (§9).

---

## 5. Raw / Unique / Shared construction (reuse accepted semantics)

Constructed per (source, model, side, layer) from **all** eligible representations of the
construction subset, using the exact accepted A4/A5 semantics in the exact space that will be
steered:

- **Raw:** `mu_E = mean` over group-A representations, `mu_M = mean` over group-B;
  `v_raw = normalize(mu_E − mu_M)` (A4 §3). Encoder and decoder.
- **Unique / Shared:** stream the uncentered second moments `C_A = H_A H_Aᵀ/n_A`,
  `C_B = H_B H_Bᵀ/n_B` (float64), eigendecompose to `U_A, U_B` (top `r_eff`), principal-angle
  SVD `U_Aᵀ U_B = P Σ Qᵀ`, then A5 §3 scores
  `s_unique(i) = E_A(i)(1−σ_i²)`, `s_shared(i) = σ_i² √(E_A(i)E_B(i))`, with `i_unique ≠
  i_shared`; signs frozen by A5 §4. Encoder and decoder.

Use ALL eligible representations across the frozen construction subset (never per utterance).
Persist directions + per-layer diagnostics (A5 §13), one independent construction per source.

---

## 6. Conditioning-CS (decoder only — FROZEN)

Paired analysis passes on the **same source construction utterances** under two language
conditions, using the accepted A4/A5 conditioning machinery (Whisper 4-token forced prefix
`<|en|>` vs `<|zh|>`; Qwen `language English<asr_text>` vs `language Chinese<asr_text>`,
`context=""`), teacher-forced on the **same reference transcript** in both passes so positions
align. Per source corpus `S ∈ {CS, ASCEND}` and decoder layer `l`, for matched
transcript-content hidden states:

```
delta_(u,t,l) = h_(u,t,l)(c_E) − h_(u,t,l)(c_M)
v_cond_cs^(S,l) = normalize( mean of delta over EMBEDDED-language / code-switching
                             transcript positions, aggregated at dataset level )
```

Pooling is restricted to **embedded-English / CS transcript positions only** (the group-A
decoder positions), then dataset-level aggregation (dialogue-balanced for CS, per the A4
aggregator; utterance-then-corpus for ASCEND). This is the **new** CS-restricted conditioning
variant — distinct from A4 Conditioning, which pooled all content positions.

---

## 7. Conditioning-All (decoder only — FROZEN)

Same paired passes as §6, but pooled over **all eligible transcript-content positions**:

```
v_cond_all^(S,l) = normalize( mean of delta over ALL eligible transcript-content positions )
```

Excluded from pooling: prompt/language-control tokens (Whisper 4-token forced prefix; Qwen
`language …<asr_text>` control tokens), audio-prefix/placeholder states, BOS/EOS where not
transcript content, padding, and all special tokens (A4 §1/§7 exclusions verbatim).

`v_cond_all^(CS,l)` reproduces the frozen A4 canonical **Conditioning** direction construction
(all-position, dialogue-balanced) at each decoder layer; `v_cond_all^(ASCEND,l)` is the same
construction on the ASCEND source. Conditioning is **decoder-only** in both variants.

---

## 8. Cross-source geometry (a central A6-F result — FROZEN)

For every **same-model, same-layer** direction, compute the cross-source cosine:

- `cos(v_raw^CS, v_raw^ASCEND)`
- `cos(v_unique^CS, v_unique^ASCEND)`
- `cos(v_shared^CS, v_shared^ASCEND)`
- decoder additionally: `cos(v_cond_cs^CS, v_cond_cs^ASCEND)`,
  `cos(v_cond_all^CS, v_cond_all^ASCEND)`

Also preserve all **within-source** geometry from A5 §13 (`cos(unique, shared)`,
`cos(unique, Raw)`, `cos(shared, Raw)`, `cos(·, Conditioning)`, projection energies). All
geometry is **within-model only** — never compare coordinates across Whisper (1280-d) and Qwen
(2048-d). Cross-source cosine tests whether the code-switching direction is corpus-portable at
the geometric level (a prerequisite for behavioral transfer).

---

## 9. Localization — Oracle-local ONLY (FROZEN)

All A6-F interventions are Oracle-local; **no Global**.

- **Encoder:** edit only oracle **embedded-English acoustic frames**.
- **Decoder:** edit only oracle **embedded-English generation/transcript positions**.
- Reuse the **final accepted A4/A5 Oracle-local masks** (Whisper A3-derived spans mapped to
  each model's frame grid; **repaired Qwen local masks**, CS `cs_ddev_select_alignment_v1`,
  SEAME frozen target segments; ASCEND masks from the canonical alignment on `ASCEND-eval`).
  Never edit prompt/language-control, audio-prefix/placeholder, padding, BOS/EOS/special
  positions; the local mask is a strict subset of the eligible transcript/frame set (A4
  acceptance checks). **Do NOT invent a new localization implementation.**

Oracle-local is an **upper bound**, not deployable inference — stated in every A6-F claim.

---

## 10. Evaluation panels (FOUR — FROZEN)

Every A6-F direction (both construction sources) is evaluated on **all four** panels:

| Panel | N | Role |
|---|---:|---|
| CS-Dialogue `D-dev-select` | 300 | evaluation |
| SEAME `dev-man` | 50 | evaluation (exploratory, underpowered) |
| SEAME `dev-sge` | 50 | evaluation (exploratory, underpowered) |
| `ASCEND-eval` | 300 (or all eligible; `BASIS_A6_ASCEND_DATA_SPEC §7`) | evaluation |

`D-dev-confirm` DO NOT USE. `D-test` DO NOT USE. ASCEND `test` DO NOT USE. No resampling.

---

## 11. Dose + decoding (FROZEN)

- **Dose:** `ρ ∈ {0.5, 1, 2, 4, 6}` — run **all** values. `ρ = 4, 6` are high-dose stress
  tests; destructive results are **kept**, not removed.
- **Decoding:** two frozen conditions, `greedy` and `official_standard`
  (`BASIS_A6_EXPANDED_SPEC §Decoding`; Whisper `official_standard = beam-5`, Qwen
  `official_standard ≡ greedy` per its shipped `generation_config` — not invented).
- NormPreserve, no depth rescale. Per-cell cross-model fairness logging (A4 §9): original
  hidden norm, mean perturbation norm, total intervention energy, relative perturbation, edited
  count, edited fraction. `ρ` is **not** treated as physically dose-matched across
  architectures.

---

## 12. Metrics + diagnostics (reuse canonical)

Canonical A4/A5 metrics: `MER`, `PIER`, `EN-WER`, matrix `CER`, `poi_corrections`,
`poi_corruptions`, `poi_net_utility = corrections − corruptions`, `outside_harm`,
`matrix_retention`, `embedded_retention`, insertions/deletions/substitutions,
`edited_positions_or_frames`, `total_intervention_energy`. A6 diagnostics: relative
perturbation, angular displacement, changed-transcript rate, target-logit margin where valid,
high-dose failure signals. No new metric implementation; the `poi_net_utility` identity must
hold for every accepted row.

---

## 13. Per-direction provenance (FROZEN)

Each A6-F direction has **one vector hash per (source, model, side, layer, method)**. Record:
construction source, model revision, site hash, mask hash, construction-config hash, panel
fingerprint, rank/`r_eff`, and the A5 diagnostics (§8). Each evaluation result row references
the single direction hash for its (source, model, side, layer, method) plus the dose, decode
mode, and panel fingerprint. Duplicate scientific keys = 0; empty provenance = 0.

---

## 14. Expected matrix (A6-F — DO NOT REDUCE)

Per construction regime (one source), direction-sites:

- **Whisper:** 4 non-conditioning methods × 64 layer-sites = **256**; 2 conditioning × 32
  decoder = **64**; total **320**. × 5 ρ × 2 decode × 4 datasets = **12,800** cells.
- **Qwen:** 4 × 52 = **208**; 2 × 28 = **56**; total **264**. × 5 × 2 × 4 = **10,560** cells.
- **One regime = 23,360 cells.**

A6-F has **two construction sources** (CS, ASCEND):

```
A6-F = 2 × 23,360 = 46,720 configuration cells.
```

These are configuration counts. **Do not reduce the requested matrix.** (Compute reuse — e.g.
Qwen `greedy ≡ official_standard` baseline/hypothesis collapse — is a caching optimization in
the execution plan, not a matrix reduction; the configuration cells remain enumerated.)

---

## 15. Key scientific questions (A6-F portion)

A. `CS-fixed → CS / SEAME / ASCEND` (within + transfer). B. `ASCEND-fixed → ASCEND / CS /
SEAME`. C. CS-fixed vs ASCEND-fixed **geometry** (§8). D. CS-fixed vs ASCEND-fixed
**behavior**. F. Whisper vs Qwen. G. ρ dose response. H. greedy vs official-standard. I.
Add-Unique vs Raw. J. Cond-CS vs Cond-All. K. encoder vs decoder. (E — fixed vs oracle
per-sample — is answered jointly with A6-TT.) Answers are **not** forced positive; a null
transfer or a reparameterization null is a valid, reportable outcome.

---

## 16. Scientific risks

1. **Cross-corpus transfer may fail** (directions are corpus-specific). Mitigation: report
   geometry (§8) and behavior separately per source; transfer failure is a result.
2. **Reparameterization null** (A6-US ≈ Raw behaviorally, high `cos(unique,Raw)`; carried from
   A5 Q9). Mitigation: report the A6-US-vs-Raw delta explicitly; do not spin a null.
3. **Cross-model / cross-corpus over-reading.** Mitigation: within-model geometry only;
   normalized relative depth for Whisper↔Qwen pattern comparison; SEAME (n=50) exploratory.
4. **High-dose destruction** at ρ=4,6. Mitigation: keep destructive results; report high-dose
   failure signals.
5. **Comparator/site/mask drift.** Mitigation: strict per-cell A4/A5 hash/site/mask/panel
   match; masks reused, not reinvented.
6. **ASCEND construction-population imbalance** vs CS. Mitigation: size-matched utterance
   count; report duration/frame/position budgets, never equalize by tuning.

---

## 17. Protocol Status: **PASS**

Construction sources, direction semantics, conditioning variants, sites, masks, panels, dose,
decoding, matrix (46,720), provenance, metrics, and comparisons are determined and frozen.
Compute in `BASIS_A6_EXECUTION_PLAN.md`. No GPU work is launched by this stage.
