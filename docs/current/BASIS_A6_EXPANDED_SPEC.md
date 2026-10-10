# BASIS-A6 — Cross-Corpus Fixed and Oracle Test-Time Local Steering (EXPANDED SPEC)

Status: **FROZEN PROTOCOL — umbrella spec for the expanded BASIS-A6 study.** This is the
authoritative entry point; the two branches and the data path are specified in full in the
companion documents and are **not** duplicated here beyond the shared contract. It reuses the
BASIS-A4 sites/masks/metrics/baseline semantics and BASIS-A5 Unique/Shared construction
**verbatim**, preserves all previous results, and adds two scientifically distinct branches, a
second construction source (ASCEND), a fourth evaluation panel, a dose sweep, and two decoding
conditions. **No GPU jobs and no scientific inference are launched by this stage.**

Companions (authoritative for their branch):
- `BASIS_A6_FIXED_SPEC.md` — **A6-F** Corpus-Level Fixed Steering.
- `BASIS_A6_ORACLE_TT_SPEC.md` — **A6-TT** Oracle Per-Sample Test-Time Steering (upper bound).
- `BASIS_A6_ASCEND_DATA_SPEC.md` — ASCEND download, adapter, subset freeze.
- `BASIS_A6_EXECUTION_PLAN.md` — compute / cache / Slurm plan (no runs launched now).

Base state: BASIS-A5 COMPLETE (`results/basis_a5_unique_shared/FINAL_MANIFEST.json`,
`status=COMPLETE`, protocol_hash `sha256:39c3f62c…dfb2`), BASIS-A4 accepted `d00c54c`.

---

## 1. The two branches

- **A6-F — Corpus-Level Fixed Steering.** One frozen direction per (construction source ×
  model × side × layer × direction type), built from **many** construction utterances, then
  evaluated **unchanged** on four panels. Tests within-corpus steering, cross-corpus transfer,
  and direction-source dependence. **Two construction sources:** `CS-D-construct`,
  `ASCEND-construct`.
- **A6-TT — Oracle Per-Sample Test-Time Steering.** For each evaluation utterance, the
  direction is built **from that same input** using oracle **location** information only (never
  gold token identity), applied to that same input, and decoded. No parameter update, no
  gradient, no learned adaptation, no corpus-level direction. Described everywhere as an
  **UPPER BOUND**, not deployable inference.

---

## 2. Models (verify from frozen config — FROZEN)

- **Whisper-large-v3:** encoder **E0–E31** (32), decoder **D0–D31** (32), `d_model = 1280`.
- **Qwen3-ASR-1.7B:** audio-encoder **E0–E23** (24, d=1024), text-decoder **D0–D27**
  (28, d=2048).

Frozen A4/A5 counts (BASIS_A4_SPEC §4–§5). Directions live in the exact hidden space of their
model/side; Whisper and Qwen spaces are never mixed.

---

## 3. Datasets (FROZEN)

Evaluation panels (both branches, all four):

| Panel | N | Role |
|---|---:|---|
| CS-Dialogue `D-dev-select` | 300 | evaluation |
| SEAME `dev-man` | 50 | evaluation (exploratory) |
| SEAME `dev-sge` | 50 | evaluation (exploratory) |
| `ASCEND-eval` (CAiRE/ASCEND `validation`) | 300 or all eligible | evaluation |

Construction sources (A6-F): `CS-D-construct` (125 utts / 20 dialogues), `ASCEND-construct`
(ASCEND `train`, size-matched). Existing datasets/splits are **unchanged**. `D-dev-confirm`,
`D-test`, and ASCEND `test` are **DO NOT USE**. ASCEND details:
`BASIS_A6_ASCEND_DATA_SPEC.md`.

---

## 4. The six steering methods (FROZEN)

| ID | Name | Direction | Sides |
|---|---|---|---|
| **A6-U** | Add-Unique | `d = +v_unique` | encoder + decoder |
| **A6-S** | Minus-Shared | `d = −v_shared` | encoder + decoder |
| **A6-US** | Unique-minus-Shared | `d = normalize(v_unique − v_shared)` | encoder + decoder |
| **A6-R** | Raw | `d = normalize(mu_E − mu_M)` | encoder + decoder |
| **A6-C-CS** | Conditioning-CS | `v_cond_cs` (CS-region pooled paired-delta) | **decoder only** |
| **A6-C-ALL** | Conditioning-All | `v_cond_all` (all-content pooled paired-delta) | **decoder only** |

Raw/U/S/U-S: encoder + decoder. Conditioning-CS / Conditioning-All: decoder only. **No Global
A6 steering.** Construction semantics reuse A4 (Raw, Conditioning-All) and A5 (Unique/Shared)
verbatim; Conditioning-CS is the new CS-region-restricted conditioning variant. Branch specs
give the exact fixed (A6-F §5–§7) and per-sample (A6-TT §4–§6) definitions.

---

## 5. Localization — Oracle-local ONLY (FROZEN)

Every A6-F and A6-TT intervention is Oracle-local. Encoder: only embedded-English acoustic
frames. Decoder: only embedded-English generation/transcript positions. Reuse the final
accepted A4/A5 localization and the **repaired Qwen local masks**; never edit
control/prefix/audio-placeholder/special/padding positions. **No Global steering.** Oracle-local
is an upper bound, not a deployable localizer — stated in every claim.

---

## 6. Dose (FROZEN)

Both branches: `ρ ∈ {0.5, 1, 2, 4, 6}` — run **all** values. `ρ = 4, 6` are high-dose stress
tests. **Destructive results are not removed.** NormPreserve, no depth rescale; per-cell
cross-model fairness logging (A4 §9). `ρ` is not treated as physically dose-matched across
architectures.

---

## 7. Decoding conditions (FROZEN — created here per §23 of the task)

No prior authoritative A6 decoding spec existed; this section **creates** it from the official
model sources and the project's DG-08 precedent (`DG08_LOCKED_WHISPER_EVAL_SPEC §65–68`, which
already froze greedy + beam-5 for Whisper). Two frozen decoding conditions for **both**
branches:

### 7.1 `greedy`
- **Whisper:** `do_sample=False, num_beams=1, temperature=0.0, task=transcribe, language=zh,
  condition_on_prev_tokens=False, max_new_tokens=200`, shipped `suppress_tokens` /
  `begin_suppress_tokens`, `return_timestamps=False` (Whisper `generation_config.json`).
- **Qwen:** Transformers backend, `do_sample=False, temperature≈0` (per
  `Qwen3-ASR-1.7B/generation_config.json`: `do_sample=false, temperature=1e-6`), `num_beams=1`,
  `force_language="Chinese"`, `context=""`, `max_new_tokens=200`, `eos=[151643,151645]`,
  `pad=151643`.

### 7.2 `official_standard`
- **Whisper:** identical to greedy except **`num_beams=5`** (beam search), matching the DG-08
  frozen "official/robust" secondary decode. No beam-specific reselection or retuning; same
  suppress/begin-suppress tokens, `language=zh`, `max_new_tokens=200`.
- **Qwen:** Qwen3-ASR ships **no beam configuration** in its `generation_config.json`; its
  canonical/official decode is greedy (`do_sample=False, temperature≈0`). Therefore **Qwen
  `official_standard ≡ Qwen greedy`.** Per the task, we **do not invent** a different Qwen
  setting merely to make the modes distinct. Both Qwen decode cells are retained in the
  configuration matrix (§8) but are **provenance-identical**; the execution plan hash-collapses
  them (a caching optimization, not a matrix reduction). This collapse is reported honestly in
  every Qwen `official_standard` row.

Exact decoding semantics/provenance (backend, config hashes, token sets) are recorded per cell.

---

## 8. Exact matrix (FROZEN — DO NOT REDUCE)

Per **one construction regime**, direction-sites = (4 non-conditioning × layer-sites) +
(2 conditioning × decoder layers):

- **Whisper:** 4 × 64 = 256; 2 × 32 = 64; total **320**. × 5 ρ × 2 decode × 4 datasets =
  **12,800**.
- **Qwen:** 4 × 52 = 208; 2 × 28 = 56; total **264**. × 5 × 2 × 4 = **10,560**.
- **One regime = 23,360 cells.**

Branch totals:

| Branch | Regimes | Configuration cells |
|---|---|---:|
| **A6-F** | 2 construction sources (CS, ASCEND) | **2 × 23,360 = 46,720** |
| **A6-TT** | 1 (per-sample) | **23,360** |
| **Total requested steering configuration cells** | | **70,080** |

Shared baselines: 2 models × 4 datasets × 2 decode modes = **16** (unsteered free-decode).

These are **configuration** counts. A6-TT U/S/U-S rows additionally carry sample eligibility
coverage (A6-TT §7). **Do not reduce the requested matrix.** Compute reuse (Qwen decode-mode
collapse; per-sample direction caching across ρ) is documented in the execution plan and does
**not** change these enumerated counts.

---

## 9. Metrics (FROZEN — reuse canonical)

`MER`, `PIER`, `EN-WER`, matrix `CER`, `poi_corrections`, `poi_corruptions`, `poi_net_utility`,
`outside_harm`, `matrix_retention`, `embedded_retention`, `edited_positions_or_frames`,
`total_intervention_energy`. A6 diagnostics: relative perturbation, angular displacement,
changed-transcript rate, target-logit margin where valid, high-dose failure signals. A6-TT
additionally: eligibility coverage, direction-construction latency, extra analysis passes, total
inference latency / RTF. No new metric implementation; `poi_net_utility` identity holds for
every accepted row.

---

## 10. Key scientific comparisons (FROZEN)

A. CS-fixed → CS / SEAME / ASCEND. B. ASCEND-fixed → ASCEND / CS / SEAME. C. CS-fixed vs
ASCEND-fixed **geometry**. D. CS-fixed vs ASCEND-fixed **behavior**. E. **fixed (A6-F) vs
oracle per-sample (A6-TT)**. F. Whisper vs Qwen. G. ρ dose response. H. greedy vs
official-standard. I. Add-Unique vs Raw. J. Cond-CS vs Cond-All. K. encoder vs decoder.

**Headline (A6-TT §16):** does per-sample Qwen steering become substantially stronger than
corpus-level Qwen steering? YES ⇒ strong input-dependent directions; NO (with real perturbation
magnitude) ⇒ evidence toward intervention-site / causal-pathway limitations. **Do not force
either conclusion.** Nulls (transfer failure, reparameterization, upper-bound-without-gain) are
valid, reportable outcomes.

---

## 11. Provenance (FROZEN)

- **A6-F:** one vector hash per (source, model, side, layer, method); result rows reference it +
  dose + decode mode + panel fingerprint (A6-F §13).
- **A6-TT:** per-sample direction records (utterance_id, method, model, side, layer, decode
  analysis mode, n_A, n_B, rank, direction hash, oracle alignment hash) → a deterministic
  `DIRECTION_BUNDLE_HASH` per (model × dataset × side × method × decode mode); every aggregate
  row references the bundle hash (A6-TT §11).
- Every contract run writes a manifest (resolved config, environment, git state, model
  metadata, hashes, status) per AGENTS.md. Duplicate scientific keys = 0; empty provenance = 0.

---

## 12. Preservation + invariants

- **Preserve all previous results/docs** (A2–A5, DG-00–DG-08). A6 is additive; it re-decodes
  nothing from prior stages and reuses A4/A5 comparators read-only.
- Scientific invariants honored: frozen DG-02 Whisper site; frozen Qwen self-attn-residual
  sites; NormPreserve; no `sqrt(num_layers)` rescale; Oracle-local only; no intervened `D-test`;
  data-role separation (`DATA_EXPOSURE.md`, incl. the new ASCEND rows).

---

## 13. Documents in this freeze

`BASIS_A6_EXPANDED_SPEC.md` (this), `BASIS_A6_FIXED_SPEC.md`, `BASIS_A6_ORACLE_TT_SPEC.md`,
`BASIS_A6_ASCEND_DATA_SPEC.md`, `BASIS_A6_EXECUTION_PLAN.md`; `scripts/download_ascend.py`;
`DATA_EXPOSURE.md` updated with ASCEND roles. **Protocol committed before implementation/runs.**

---

## 14. Protocol Status: **PASS**

Branches, models, datasets, six methods, localization, dose, decoding conditions, exact matrix
(46,720 + 23,360 = 70,080; baselines 16), metrics, comparisons, and provenance are determined
and frozen. Branch and data specifics are authoritative in the companion documents. No GPU work
and no scientific inference are launched by this stage.
