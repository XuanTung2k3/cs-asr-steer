# BASIS-A4 — Final Independent Audit & Scientific Synthesis

Auditor stance: independent, read-only. No GPU jobs, no reruns, no scientific-setting
changes. Audited against `docs/current/BASIS_A4_SPEC.md` + `BASIS_A4_EXECUTION_PLAN.md`
(protocol commit `aeafa03`). This audit reports the **actual on-disk state**, not the
planned matrix, and does **not** fabricate results for cells that were never decoded.

---

## Final Status: **BLOCKED**

The A4 core matrix is materially incomplete and two required pre-GPU gates are **FAIL**:

1. **Qwen3-ASR arm is entirely absent.** No construction, no baselines, no encoder/decoder
   atlas cells (0 of ~480 planned steered cells + construction + 3 baselines). Qwen CPU
   acceptance **FAILED** and the Qwen preflight reports **FAIL**.
2. **Whisper Conditioning on the only powered panel (CS-Dialogue, n=300) is incomplete**
   — 7 of 32 decoder layers present (L0,L1,L2 + reused L24/26/27/31); 25 layers (50 cells)
   missing. Jobs `52083`/`52085` are stuck `RUNNING` (never finalized; disk at 97%).
3. A **reuse/fresh double-representation** of SEAME Conditioning L24/26/27/31 must be
   reconciled before any aggregate.

Cross-model synthesis (the headline scientific goal of A4) is therefore **impossible** —
only one model produced data. What is valid is reported below, strictly scoped.

---

## Completeness

Protocol freeze present and self-consistent (`manifests/a4_protocol_freeze.json`:
directions `[Raw, Conditioning]`, Conditioning-Avg `REDUNDANT/run=false`, rho 0.5,
panel fingerprints match A3, `norm_preserve=true`, `depth_rescale=false`).

| Arm | Expected (rho=.5) | On disk | Status |
|---|---|---|---|
| Whisper Raw enc E0–E31 ×2×3 | 192 | 192 reused | ✅ complete (reuse) |
| Whisper Raw dec D0–D31 ×2×3 | 192 | 192 reused | ✅ complete (reuse) |
| Whisper Conditioning CS-Dialogue D0–D31 ×2 | 64 | 14 (L0,1,2 + reused L24,26,27,31) | ❌ **25 layers / 50 cells missing** |
| Whisper Conditioning SEAME-man D0–D31 ×2 | 64 | 64 | ✅ complete |
| Whisper Conditioning SEAME-sge D0–D31 ×2 | 64 | 64 | ✅ complete |
| Whisper Conditioning-Avg | 0 | 0 | ✅ correctly not materialized (REDUNDANT) |
| Qwen Raw enc E0–E23 ×2×3 | 144 | 0 | ❌ absent |
| Qwen Raw dec D0–D27 ×2×3 | 168 | 0 | ❌ absent |
| Qwen Conditioning dec D0–D27 ×2×3 | 168 | 0 | ❌ absent |
| Qwen Conditioning-Avg | 0 | 0 | ✅ N/A (REDUNDANT) |
| Qwen direction construction | 1 pass | 0 | ❌ absent |
| Qwen baselines (3 panels) | 3 | 0 | ❌ absent |

Reuse manifest: **408 accepted, 0 rejected, unique keys** (384 Raw + 24 Conditioning
L24/26/27/31 across 3 datasets). Conditioning-Avg `REDUNDANT_NOT_PRESENT`.
Whisper new-decode completed: **SEAME-man 64 + SEAME-sge 64 + CS 6 = 134**; missing CS
new-decode: **50**. Qwen new-decode: **0 / 480**.

**Blocking gates:**
- `manifests/acceptance_cpu.json` → `status: FAIL`, `required_before_gpu: true`. Failing
  test `tests/test_basis_a4_qwen.py::test_qwen_decoder_control_positions_and_cache_positions`
  → `AttributeError: '_Bundle' object has no attribute 'thinker'` in
  `src/csasr/lss/qwen_sites.py:261`. The Qwen text-decoder hook is **not validated**.
- `manifests/qwen_preflight.json` → `status: FAIL` (peak VRAM 3.86 GB, so MIG is fine;
  `cached_encoder_identity=false` among otherwise-passing checks). Qwen atlas correctly
  **not** launched while the gate is red.

---

## Protocol Validity

Audited on the cells that exist (Whisper):

- **Direction definitions** — Conditioning cells carry `direction_hash`, `rho=0.5`,
  `scale` (per-layer s_l), site implied by stage; Raw reuse cells carry A3
  `source_direction_hash` + `run_manifest` pointer. ✅
- **Vector provenance / no SEAME construction leakage** — all vectors built on CS-Dialogue
  D-construct; SEAME panels appear only as evaluation panels (`data_role` SEAME-dev-man/sge).
  No SEAME-constructed vector found. ✅
- **Intervention site exactness** — Whisper decoder `decoder_post_cross_attn_residual`
  (DG-02 FROZEN), encoder `encoder_post_self_attn_residual_pre_ffn`. Qwen sites frozen in
  spec but **unverified** (acceptance hook test fails). ⚠️ Qwen only.
- **Oracle-local masks** — both scopes present per completed cell; SEAME local reuses the
  A3 target-segment-union mask. ✅ (Whisper)
- **Prompt/audio-prefix exclusions** — inherited from A3 decoder-global eligibility. ✅
- **NormPreserve / no depth rescale** — `norm_preserve=true`, `depth_rescale=false` in
  freeze; Qwen preflight confirms `norm_preserve_nonzero=true`. ✅
- **rho=0 identity / no gradients** — frozen from DG-02 for Whisper; Qwen preflight
  confirmed `decoder_rho0_identity`, `encoder_rho0_identity`, `no_gradients` = true. ✅
- **Metric semantics** — cells expose canonical `metrics_v1` incl. `poi_corrections`,
  `poi_corruptions`, `poi_net_utility`, retention, MER/PIER/EN-WER/ZH-CER + gains, and the
  fairness logs (`mean_intervention_norm`, `total_intervention_energy`,
  `edited_positions_or_frames`). ✅
- **Hashes / duplicate keys / quarantine** — reuse keys unique; `quarantine/whisper_
  conditioning_legacy_layout/` (25 files) correctly excluded. ⚠️ **Reconciliation defect:**
  SEAME-man/sge L24/26/27/31 exist BOTH as accepted reuse (manifest) AND as freshly
  decoded A4 cells → potential double-count; finalize must dedupe to one source.
- **Model revision** — Qwen revision recorded in preflight; Whisper inherits A3 model
  metadata via reuse `run_manifest`. ✅
- **Data exposure** — only CS `D-construct` (construction) + CS `D-dev-select` (eval) +
  SEAME dev-man/sge (eval). No `D-dev-confirm`/`D-test`. ✅
- **Provenance gap** — new Whisper Conditioning cells carry `provenance: {}` (empty);
  full model/env/hash provenance lives only in job manifests, and jobs ran with
  `git.dirty=true` (untracked `experiments/basis_a4.py`, `src/csasr/lss/qwen_sites.py`,
  etc.). Must be committed before any claim. ⚠️

---

## Whisper Findings

All Whisper Conditioning numbers below are POI `net(corr/corru)`. **SEAME panels are n=50
(underpowered, EXPLORATORY).** CS-Dialogue Conditioning sweep is **incomplete** (7/32
layers); CS Raw is reused from A3 (n=300).

**Whisper Raw decoder, CS-Dialogue (n=300, reused A3):** early layers damaging under global
(L0 −54, L4 −49, L8 −31), turning net-positive from the mid-decoder (L12 +43, L16 +89) and
peaking in a broad **late band L26–L31 global** (L27 +111, L29 +144, L30 +151, L31 +109),
corrections dominating corruptions. Oracle-local is consistently safer but lower-yield
(positive but smaller), i.e. localization trades corrections for sharply fewer corruptions.

**Whisper Conditioning, CS-Dialogue (n=300, partial):** L0 global +48(83/35); L1/L2
net-negative; the mid anchors are damaging under global and rescued by localization
(L24 G −172(39/211) → L −59(27/86); L26 G −69 → L −2; L27 G −21 → L +5); **L31 is a strong
late language-control layer** — global +126(459/333) but **local +361(395/34)**, i.e.
localization cuts corruptions 333→34 while keeping ~395 corrections. The would-be peak band
(L28–L30), by analogy to Raw and SEAME, is exactly what is **missing** on CS.

**Whisper Conditioning, SEAME (n=50, complete, both scopes):**
- dev-man global: positive early (L0 +5, L4 +3), a **damaging mid/late band L7–L27**
  (L22 −7, L23 −8, L24 −9), then a **strong late regime L28–L30** (L28 +7, L29 +12(14/2),
  L30 +5); L31 global mixed (+−, many corr & corru).
- dev-man **oracle-local is robustly positive and flat** (~+3, corruptions ≈0) across almost
  all layers — localization converts the destructive global mid-band into safe gains.
- dev-sge global: mostly flat/negative mid-depth with a **late positive regime L28–L30**
  (L28 +10, L29 +11, L30 +8); local ≈0 (few embedded-English POIs to correct in sge).

## Qwen Findings

**NO DATA.** Construction, baselines, and all atlas cells are absent; the CPU acceptance
and preflight gates are FAIL. No Qwen scientific statement can be made. The only verified
Qwen facts are architectural/operational (from the preflight + config): decoder dim 2048 /
28 layers, encoder dim 1024 / 24 layers, bf16, peak VRAM ≈3.9 GB (MIG-eligible), language
conditions produce distinct states (`language_condition_max_abs_diff≈0.92`,
`states_distinct=true`), rho=0 identity and no-gradients hold on the tested row — but the
text-decoder hook fails its control-position test, so **site exactness is unverified**.

---

## Raw

Whisper only (reused A3): useful **decoder** region is the late band (≈L26–L31 global) with
a mid-decoder onset (~L12–L16); early decoder layers are damaging under global. Encoder Raw
was not re-examined here (reused). Global yields more corrections but more corruptions;
oracle-local is safer, lower-yield. Cross-corpus: the CS late-band signal is echoed by the
SEAME late-layer positive regime (below), but on n=50. **Qwen Raw: no data.**

## Conditioning

Whisper only: a **late-layer language-control regime** is the robust motif — CS L31 (powered)
and SEAME L28–L30 (underpowered) are where global Conditioning becomes strongly corrective;
mid-depth global Conditioning is damaging. **Localization dependence is strong:** oracle-local
removes most corruption (CS L31 corru 333→34; SEAME-man mid-band flips negative→+3). The
effect is **broad-but-signed** (many layers engaged, sign depends on depth×scope), not narrow.
**Qwen Conditioning: no data** → no cross-model confirmation of the late-control regime.

## Conditioning-Avg

**Confirmed REDUNDANT — not a genuinely different population/direction.** The A4 task's
premise (that Conditioning and Conditioning-Avg are distinct populations) is **false under
the traced construction**: the existing Conditioning already averages the English−Mandarin
conditioning contrast over **all** transcript-content positions (matrix + embedded), with
the identical eligible-position set, subtraction order, normalization, and per-layer scheme
(`BASIS_A4_SPEC §1–2`; `dg03_build_basis._conditioning_and_scale`). The only difference is
cross-utterance aggregation weighting (dialogue-balanced utterance-mean vs token-weighted
grand mean).

- **Layerwise cos(Cond, CondAvg):** ≈**1.0 by construction** at every layer (same population,
  same contrast; only a re-weighting of equal-valued per-utterance terms). Not computed
  numerically because CondAvg was correctly **not materialized** — computing it would require
  building a duplicate vector the gate forbids.
- Does CondAvg approximate Cond / Raw / remain distinct? → **approximates Cond (it *is* Cond
  up to weighting)**; it does **not** approximate Raw (Raw is a separate baseline-correct
  positional contrast). Questions about CondAvg's stability / breadth / destructiveness /
  SEAME transfer (task §4) are **moot** — there is no distinct direction to characterize.

## Global vs Local

Whisper, consistent across CS (powered, partial) and SEAME (n=50): **Global = higher
corrections + higher corruptions; Oracle-local = far fewer corruptions, modest corrections.**
Localization is most valuable exactly where global is destructive (mid/late decoder). This is
a diagnostic **upper bound**, not a deployable localizer. **Qwen: no data.**

## Encoder vs Decoder

Decoder results only (Whisper Raw reused; Conditioning decoder-only by design). Encoder Raw
is reused (not re-audited). **No Qwen encoder data**, so the planned "encoder failure
reproduces across models" question is **unanswerable**.

## Cross-Corpus Generalization

Within Whisper: the **late-layer corrective regime reproduces** CS→SEAME (CS L31 / SEAME
L28–L30), and the **mid-depth global-damage + localization-rescue** pattern reproduces on
SEAME-man. But SEAME panels are n=50 and dev-sge has few correctable embedded POIs. CS
Conditioning's own late band (L28–L30) is **missing**, so the strongest cross-corpus claim
cannot yet be anchored on the powered panel. **EXPLORATORY.**

## Cross-Model Generalization

**UNSUPPORTED — no Qwen data.** Every cross-model question (relative-depth useful band, Raw
encoder failure, late steering stronger-but-more-damaging, localization value at depth,
late language-control regime in both models, CondAvg transfer) is **unanswerable**. Do not
claim hidden-space equivalence; do not claim rho=0.5 dose-equivalence. A4's defining
deliverable is not met.

## Geometry

No geometry artifacts were produced (`results/basis_a4/geometry/` absent). Within-model
Raw↔Conditioning geometry (CPU) remains to be computed from existing Whisper vectors; it is
not blocked by GPU. Cross-model geometry is forbidden (different hidden spaces) and not at
issue.

---

## Strong Claims

- **STRONG (methodological):** Conditioning-Avg is **REDUNDANT** with Conditioning; the third
  direction collapses to a weighting variant. (Code-traced, not data-dependent.)
- **STRONG (powered, reused A3):** On CS-Dialogue (n=300), **Whisper Raw decoder steering has a
  useful late band (≈L26–L31) under global scope**, with early-decoder global steering
  damaging. (This is an A3 result reused, not new A4 evidence.)

## Moderate / Exploratory Claims

- **MODERATE (powered, partial):** Whisper **Conditioning at L31 shows a strong language-control
  effect dominated by localization** (CS local +361 vs global +126; corruptions 333→34).
  Limited by the incomplete CS sweep.
- **EXPLORATORY (n=50):** Whisper Conditioning exhibits a **late-layer corrective regime
  (L28–L30)** and a **destructive mid-band rescued by oracle-local** on SEAME-man; weaker,
  late-only positives on SEAME-sge.
- **EXPLORATORY:** Global-vs-local corruption trade-off is consistent across available panels.

## Claims to Avoid

- Any **cross-model / Qwen** claim (no data; hooks unverified). **UNSUPPORTED.**
- Any claim that rho=0.5 is dose-equivalent across architectures.
- Any **powered CS Conditioning layer-profile** claim (sweep incomplete).
- Any deployable-localizer claim (oracle-local is an upper bound).
- Any claim from SEAME treated as confirmatory rather than exploratory (n=50).
- Direct hidden-space coordinate comparisons across models.

## Best Paper Figures (conditional on completion)

- **Mechanistic (Whisper):** depth × scope POI-net-utility heatmap for Raw (CS, n=300,
  available) and Conditioning (needs CS completion) — shows the damaging-early /
  corrective-late transition and the localization rescue.
- **Strongest single result available now:** CS L31 Conditioning global-vs-local corruption
  collapse (333→34) — motivates **adaptive/damage-aware, localized** steering (it is the
  clearest "global damage is avoidable with targeting" datapoint).
- **Cross-model figure:** relative-depth Raw useful-band overlay Whisper vs Qwen — **blocked**
  (Qwen missing).
- **Negative results worth reporting:** mid-depth global Conditioning is net-damaging;
  SEAME-sge has little correctable embedded-English headroom.
- **Appendix-only:** SEAME n=50 per-layer tables; energy/edited-fraction fairness logs.

---

## Remaining Necessary Experiment

Scientifically necessary before any A4 claim beyond the redundancy finding:

1. **Fix the Qwen text-decoder hook** (`qwen_sites.py` `thinker` attribute) so CPU acceptance
   passes; re-run the Qwen preflight to green (resolve `cached_encoder_identity`).
2. **Run the entire Qwen arm** (construction on CS D-construct; 3 baselines; Raw enc+dec and
   Conditioning dec atlas, both scopes, 3 panels) — 0/480 done.
3. **Complete Whisper Conditioning on CS-Dialogue** (25 missing layers L3–23, L25, L28–30 ×2
   scopes = 50 cells), so the powered panel has a full depth profile incl. the L28–L30 band.
4. **Reconcile the SEAME L24/26/27/31 reuse-vs-fresh double-representation** and commit the
   A4 code (currently untracked / dirty tree) before producing aggregates.
5. **Compute within-model Raw↔Conditioning geometry** (CPU; not GPU-blocked).

(Free disk first — root FS at 97%, which already stalled the CS Conditioning jobs.)

Until (1)–(3) land, the cross-model atlas that defines BASIS-A4 does not exist, so the study
is **BLOCKED**, not PASS.
