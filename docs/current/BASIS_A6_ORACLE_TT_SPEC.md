# BASIS-A6-TT — Oracle Per-Sample Test-Time Steering (SCIENTIFIC SPEC)

Status: **FROZEN PROTOCOL — oracle per-sample, test-time, upper-bound steering.** For each
evaluation utterance the steering direction is constructed **from that same input** using
oracle **location** information only, then applied to that same input and decoded. **No
external corpus-level vector, no parameter update, no gradient, no learned adaptation.** This
is an **UPPER BOUND**, explicitly **not** deployable inference. **No GPU jobs and no scientific
inference are launched by this stage.** Part of the BASIS-A6 freeze
(`BASIS_A6_EXPANDED_SPEC.md`).

Base state: BASIS-A5 COMPLETE, BASIS-A4 accepted at `d00c54c`. Companions:
`BASIS_A6_FIXED_SPEC.md`, `BASIS_A6_ASCEND_DATA_SPEC.md`, `BASIS_A6_EXECUTION_PLAN.md`.

---

## 0. Scope in one paragraph

For each evaluation utterance `x`: run a baseline/analysis pass → construct per-sample
direction(s) `v_·^(x,l)` from `x` using oracle **language-region location** only → steer that
same `x` Oracle-local → decode. Directions are **never** shared across utterances and **never**
constructed from a corpus. Evaluated on the same four panels as A6-F. The headline question:
does per-sample steering (especially on Qwen) become **substantially stronger** than
corpus-level (A6-F) steering — evidence for strong input-dependent directions — or not, which
would point more toward **intervention-site / causal-pathway** limitations. The conclusion is
**not** forced either way.

---

## 1. Models / panels / sites (FROZEN — same as A6-F)

- Models: Whisper-large-v3 (enc E0–E31, dec D0–D31, d=1280); Qwen3-ASR-1.7B (enc E0–E23 d=1024,
  dec D0–D27 d=2048).
- Panels: CS `D-dev-select` (300), SEAME `dev-man` (50), SEAME `dev-sge` (50), `ASCEND-eval`
  (300 or all eligible). `D-dev-confirm` / `D-test` / ASCEND `test` DO NOT USE.
- Sites, NormPreserve, no depth rescale: identical to A6-F §3.

---

## 2. Oracle information budget (FROZEN — location only, not identity)

**Allowed** oracle information (location only):
- true CS-region / language-region labels,
- accepted acoustic **span** alignment,
- accepted reference-to-hypothesis region alignment.

**Forbidden:** using the **gold correct token identity** to construct directions. In
particular, **do NOT teacher-force the gold reference merely to obtain a direction** if the
same construction can be performed from the model's **baseline hypothesis**. The oracle answers
*"where is the CS region?"*, never *"what is the correct target token?"*.

### 2.1 Preferred decoder analysis protocol (FROZEN 5 steps)

1. **Baseline free decode** of `x` under the SAME requested decoding mode (greedy or
   official_standard).
2. **Align** oracle language-region labels onto the **baseline-hypothesis** transcript
   positions using the accepted alignment machinery (reference↔hypothesis region alignment).
3. **Extract** analysis/teacher-forced hidden states on the **baseline-hypothesis** sequence
   (NOT the gold reference).
4. **Construct** the per-sample direction(s) from those hidden states.
5. **Steer** and perform the final oracle-local steered free decode.

Any unavoidable exception (e.g. a sample where the baseline hypothesis has no alignable
embedded region) is **documented** per utterance, not silently resolved. Encoder directions are
built directly from `x`'s encoder frames (no hypothesis needed; §3).

---

## 3. Per-sample groups (FROZEN)

For input `x` and layer `l`, no other utterance contributes.

- **Encoder (§14):**
  - **Group A:** hidden **frames** inside oracle embedded-English acoustic spans.
  - **Group B:** hidden frames inside oracle matrix-language acoustic spans.
  Built directly from `x`'s encoder pass (decoding-independent where technically exact).
- **Decoder (§15):**
  - **Group A:** hidden **positions** on the **baseline-hypothesis analysis sequence** aligned
    to oracle embedded-English regions.
  - **Group B:** positions aligned to matrix-language regions.

`H_A^(x,l) ∈ R^{d×n_A}`, `H_B^(x,l) ∈ R^{d×n_B}` are the per-sample group matrices.

---

## 4. Per-sample Raw (FROZEN)

For every eligible `x, l`:

```
mu_E^(x,l) = mean(H_A^(x,l))
mu_M^(x,l) = mean(H_B^(x,l))
v_raw^(x,l) = normalize( mu_E^(x,l) − mu_M^(x,l) )
```

Used only for `x`. Raw eligibility: `n_A ≥ 1` and `n_B ≥ 1` (means defined). Raw may therefore
have **broader** eligibility than Unique/Shared (§5); this is reported separately (§7).

---

## 5. Per-sample Unique / Shared — dynamic rank (FROZEN)

Use the A5 geometric extraction (uncentered second moments → principal angles → the A5 §3
scores and sign rules), but the available rank is **sample-dependent**:

```
r_(x,l) = min( 32, n_A, n_B, d )
```

- **Do NOT zero-pad to rank 32.** Use truncated subspaces at `r_(x,l)`.
- Scores (A5 §3): `s_unique(i) = E_A(i)(1 − σ_i²)`, `s_shared(i) = σ_i² √(E_A(i) E_B(i))`.
- Require `i_unique ≠ i_shared`.
- Signs frozen from per-sample construction geometry only (A5 §4 rules applied to `x`).

**Eligibility for U / S / U-S:** require `r_(x,l) ≥ 2` (needed to pick two distinct principal
indices and support all of Add-Unique, Minus-Shared, U-S). If `r_(x,l) < 2`, mark the
(utterance, layer) as **`INELIGIBLE_UNIQUE_SHARED`**. **Do NOT** silently replace it with Raw;
**do NOT** invent another vector. The three methods A6-U/A6-S/A6-US are simply not produced for
that (utterance, layer).

Per-sample directions: A6-U `+v_unique^(x,l)`, A6-S `−v_shared^(x,l)`, A6-US
`normalize(v_unique^(x,l) − v_shared^(x,l))`. Used only for `x`.

---

## 6. Per-sample Conditioning-CS / Conditioning-All (decoder only — FROZEN)

Paired analysis passes on the **same current input `x`** under `c_E` and `c_M`, using the
**same baseline-hypothesis transcript sequence** in both passes so positions align (this uses
the model's own hypothesis tokens, not gold identity; the only oracle input is region
location):

```
delta_(x,t,l) = h_(x,t,l)(c_E) − h_(x,t,l)(c_M)
```

- **Conditioning-CS:** `v_cond_cs^(x,l) = normalize( mean delta over oracle embedded-language
  transcript positions )`.
- **Conditioning-All:** `v_cond_all^(x,l) = normalize( mean delta over ALL eligible
  transcript-content positions )`; exclude all prompt/control, audio-prefix/placeholder,
  BOS/EOS-non-content, padding, and special positions (A4 §1/§7 exclusions).

Decoder only. Used only for `x`. Conditioning eligibility: `≥ 1` pooled position of the
required type; reported separately from U/S eligibility (§7).

---

## 7. Eligibility reporting (FROZEN)

For A6-TT, report per **(model, dataset, side, layer)** at minimum:
- total utterances,
- direction-eligible utterances,
- eligibility rate,
- `n_A` median / p10 / p90,
- `n_B` median / p10 / p90,
- `r_(x)` median / p10 / p90.

For the **U / S / U-S** comparison, provide a **`COMMON_ELIGIBLE_SUBSET`**: the set of
utterances eligible for all three at that (model, dataset, side, layer), so the three methods
are compared on **identical** utterances. Raw and Conditioning may have broader eligibility;
their coverage is reported **separately** and their aggregates are never silently mixed with the
common-subset aggregates.

---

## 8. Localization — Oracle-local ONLY (FROZEN)

Identical rule to A6-F §9: encoder edits only oracle embedded-English acoustic frames; decoder
edits only oracle embedded-English generation/transcript positions; reuse final accepted A4/A5
localization + repaired Qwen local masks; no Global; never edit control/prefix/special/padding.
Oracle-local per-sample steering is an **upper bound**, not deployable.

---

## 9. Dose + decoding (FROZEN — same axes as A6-F)

- Dose `ρ ∈ {0.5, 1, 2, 4, 6}`, all values; ρ=4,6 high-dose stress; destructive results kept.
- Decoding `greedy` and `official_standard` (`BASIS_A6_EXPANDED_SPEC §Decoding`).
- NormPreserve, no depth rescale; per-cell fairness logging (A4 §9). **Directions are
  constructed ONCE per (model, dataset, decode mode, utterance) and reused across all ρ** (§10).

---

## 10. Cache / compute design (FROZEN — construction is expensive)

Direction construction must **not** be repeated for every ρ. For each
`(model, dataset, decode mode, utterance)` perform analysis **once** and cache the per-sample
direction artifacts, then reuse them across `ρ ∈ {0.5,1,2,4,6}`.

- **Encoder Raw/U/S/U-S:** encoder hidden states are decoding-independent where technically
  exact; reuse across decode modes **iff** the input/model states and local alignment are
  identical (verified by hash). Otherwise cache per decode mode.
- **Decoder directions:** the baseline hypothesis may depend on decode mode; cache **separately
  by decode mode** unless hashes prove the hypotheses identical (e.g. Qwen greedy ≡
  official_standard → hypotheses identical → single cache shared across both modes).
- **Paired Conditioning (`c_E`/`c_M`):** cache the paired analysis states / directions once per
  required (utterance, decode mode, layer), reuse across ρ.

Only the steered **decode** varies with ρ; the direction bundle is fixed per
(model, dataset, side, method, decode mode).

---

## 11. Per-sample provenance + bundle hash (FROZEN)

A6-TT cannot use one global vector hash. For **every** per-sample direction, save:
`utterance_id`, `method`, `model`, `side`, `layer`, `decoding analysis mode`, `n_A`, `n_B`,
`rank` (`r_(x,l)`), `direction_hash`, `oracle_alignment_hash`.

Build a deterministic **`DIRECTION_BUNDLE_HASH`** for each
`(model × dataset × side × method × decode mode)` from the **ordered set of per-sample
direction hashes** (ordered by a fixed utterance-ID sort). Every aggregate A6-TT result row
**references its `DIRECTION_BUNDLE_HASH`** (plus dose, decode mode, panel fingerprint). Rows
whose bundle hash does not match the recorded per-sample set are rejected.

---

## 12. Metrics + diagnostics (reuse canonical + TT extras)

Canonical A4/A5 metrics + A6 diagnostics exactly as A6-F §12. **Additionally** report for
A6-TT:
- eligibility coverage (§7),
- direction-construction **latency**,
- number of extra analysis passes,
- total inference **latency / RTF**.

`poi_net_utility` identity holds for every accepted row; aggregates over the
`COMMON_ELIGIBLE_SUBSET` for U/S/U-S, over the broader eligible set (reported separately) for
Raw/Conditioning.

---

## 13. Expected matrix (A6-TT — DO NOT REDUCE)

A6-TT is **one construction regime** (per-sample; no external source):

```
A6-TT = 23,360 configuration cells
      = Whisper (320 direction-sites × 5 ρ × 2 decode × 4 datasets = 12,800)
      + Qwen    (264 direction-sites × 5 ρ × 2 decode × 4 datasets = 10,560).
```

where direction-sites = (4 non-conditioning × layer-sites) + (2 conditioning × decoder layers)
= Whisper (4×64 + 2×32 = 320), Qwen (4×52 + 2×28 = 264). These are **configuration** counts;
U/S/U-S rows additionally carry **sample eligibility coverage** (§7) and are compared on the
`COMMON_ELIGIBLE_SUBSET`. **Do not reduce the requested matrix.** Ineligible per-sample cells
are *recorded as ineligible* (they still occupy their configuration cell), never dropped from
the enumeration.

---

## 14. Encoder groups (detail — FROZEN)

For input `x` and encoder layer `l`: Group A = hidden frames inside oracle embedded-English
acoustic spans; Group B = frames inside oracle matrix-language acoustic spans. Constructed
directly from `x`. No external utterance contributes. Frame selection reuses the accepted A4/A5
acoustic-span→frame-grid mapping (each model's measured fps; A4 §8).

---

## 15. Decoder groups (detail — FROZEN)

For `x` and decoder layer `l`: Group A = hidden positions on the **baseline-hypothesis**
analysis sequence aligned to oracle embedded-English regions; Group B = positions aligned to
matrix-language regions. No other utterance contributes. Alignment uses the accepted
reference↔hypothesis region alignment (§2.1 step 2) — location only, not token identity.

---

## 16. Key scientific question (headline)

**Does per-sample Qwen steering become substantially stronger than corpus-level Qwen
steering?**
- **If YES:** evidence supports strong input-dependent steering directions.
- **If NO**, while the perturbation magnitude is real: evidence points more strongly toward
  **intervention-site / causal-pathway** limitations.

Do **not** force either conclusion. Comparison E (fixed A6-F vs oracle per-sample A6-TT) is the
central cross-branch analysis; F (Whisper vs Qwen) frames it per model.

---

## 17. Scientific risks

1. **Gold-identity leakage** (constructing directions from the gold reference). Mitigation: §2
   forbids it; use the baseline hypothesis; document any unavoidable exception.
2. **Small per-sample support** → many `INELIGIBLE_UNIQUE_SHARED`. Mitigation: dynamic rank
   `r_(x,l)`, explicit ineligibility marker, `COMMON_ELIGIBLE_SUBSET`, no Raw substitution.
3. **Upper-bound over-reading.** Mitigation: every A6-TT claim states "oracle per-sample =
   upper bound, not deployable."
4. **Decode-mode dependence of the hypothesis.** Mitigation: cache decoder directions per
   decode mode unless hashes prove identity (§10).
5. **Latency cost.** Mitigation: construct once per (model, dataset, decode, utterance), reuse
   across ρ (§10); report construction/inference latency + RTF (§12).
6. **Per-sample provenance sprawl.** Mitigation: `DIRECTION_BUNDLE_HASH` over the ordered
   per-sample hashes; every aggregate references it (§11).
7. **Eligibility-set confounding** (comparing methods on different utterance sets).
   Mitigation: `COMMON_ELIGIBLE_SUBSET` for U/S/U-S; Raw/Conditioning coverage reported
   separately (§7).

---

## 18. Protocol Status: **PASS**

Information budget, analysis protocol, per-sample group/definitions, dynamic rank/eligibility,
localization, dose/decoding, cache design, per-sample provenance + bundle hash, metrics, and
the matrix (23,360) are determined and frozen. Compute in `BASIS_A6_EXECUTION_PLAN.md`. No GPU
work is launched by this stage.
