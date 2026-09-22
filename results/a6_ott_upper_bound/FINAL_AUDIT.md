# A6-OTT — Independent Final Audit

Auditor stage: independent audit + freeze. **No Phase-A/B/C GPU experiment was rerun; no
scientific result file was modified; SEAME was not inspected for selection; no frozen
search/confirmation assignment was altered; no conclusion was rewritten to improve results.**
CPU-only validation was used. Audited commit **`cef08b5`** (branch `learned-expansion`).
Freeze metadata: `FINAL_FREEZE.json`.

Verdict: **PASS.** All 19 consolidated CPU assertions passed; both A6-OTT pytest suites passed
(11 tests). Scientific conclusions of the run are supported by the artifacts and are **not**
inflated.

---

## 1. Scope integrity — PASS
13,082 accepted result rows scanned (`confirm/rows` + `transfer/rows`): **all** `regime =
oracle_tt`; method families are **only** `add_unique` and `conditioning_cs`; **zero** rows from
Raw / Minus-Shared / U-S / Conditioning-All / corpus-level fixed / Global. `intervention_site`
∈ {`decoder_post_cross_attn_residual` (Whisper decoder alias), `encoder_post_self_attn_residual_pre_ffn`}.
- **Note (cosmetic, non-blocking):** the row-level `intervention_site` string is a generic alias;
  Qwen decoder rows carry it too, but `perturbation_diagnostics.records[].site` shows the correct
  `qwen_text_decoder_post_self_attn_residual_pre_mlp`. Steering is applied at the correct Qwen
  tensor; only the top-level label is generic.

## 2. Search / Confirm / Transfer separation — PASS
`CS_SEARCH=20`, `ASCEND_SEARCH=20`. `search ∩ confirm = ∅` for both corpora (verified by ID
set intersection). SEAME appears only under `transfer/`; `FINAL_MANIFEST.phase_order_assertions`
records `seame_opened_after_phase_b_freeze=true`, `phase_a_selection_read_only_search=true`.

## 3. Search-manifest repair — PASS
`preflight/SEARCH_ALIGNMENT_AUDIT_V2` (status PASS): **8** structurally-ineligible CS Search IDs
removed and **8** replacements added, each `removed_status=STRUCTURALLY_INELIGIBLE`,
`replacement_status=FULLY_ELIGIBLE`, `reason=ORACLE_ALIGNMENT_INELIGIBLE`,
`steering_outcomes_consulted=false`. Replacements are unique, all present in the final search
set, and none of the removed IDs remain. The repaired manifest (`manifests/v2/SEARCH_FREEZE.json`,
`sha256:4f011145…`) is the manifest `PHASE_A_COMPLETION` executed against; V1 (`f9d3cd97…`) is
the pre-repair predecessor. Repair was deterministic, outcome-blind, and frozen before the first
steering outcome.

## 4. Oracle information integrity — PASS
Every one of 13,082 rows: `gold_leakage.analysis_sequence = baseline_hypothesis`,
`gold_next_token_logits=false`, `gold_target_hidden_states=false`,
`target_embedding_direction=false`. Decoder directions are constructed on the baseline-hypothesis
sequence, not the gold reference. Zero gold-leakage violations (Whisper + Qwen).

## 5. Per-sample direction integrity — PASS
`provenance.direction_construction = per_sample_dynamic_rank`, `fixed_direction_artifact = null`
on all rows → directions built from the current utterance, no dataset-level Unique vector.
Physical acceptance shows `direction_sample_varying=true`. The rank-1 Add-Unique executor gap is
handled (`tests/test_a6_ott.py::test_add_unique_accepts_frozen_rank_one_case` PASS).
Conditioning-CS uses paired `c_E`/`c_M` on the baseline hypothesis, averaged over oracle CS
positions only.

## 6. Local-steering integrity — PASS
`PHYSICAL_ACCEPTANCE_{whisper,qwen}` PASS: `local_edit_nonzero=true`. Per-layer
`perturbation_diagnostics` show non-CS position groups with `active_positions=0` →
`edit_norm_sum=0` (no outside-mask edit), CS positions edited. Encoder edits only oracle acoustic
frames; decoder edits only oracle CS decoder positions. (rho=0 identity is inherited from the
frozen A4/A5 NormPreserve kernel; the A6-OTT grid has no rho=0 cell.)

## 7. Cache / all-layer reuse — PASS
`provenance.analysis_layer_collection = all_requested_layers_one_pass`;
`rho_cache = one_direction_all_rho`; **all** rows `rho_cache_reused=true`. Physical acceptance
counters: one direction construction per sample/layer, reused across rho (e.g. Whisper decoder
16 constructions → 48 rho-cache hits; Qwen 16 → 64). Paired Conditioning-CS analysis is not
rerun per rho.

## 8. Phase-A completeness — PASS
Logical = **Whisper 11,520**, **Qwen 12,800**; resolved = logical; `missing=0`, `invalid=0`;
`expected_key_hash == accepted_key_hash`; `confirmation_and_seame_outcomes_loaded=false`.
Exact-reused 72 (W) / 96 (Q); computed-new 11,448 / 12,704.

## 9. Phase-A selection validity — PASS
`selection_only_phase_a=true`, ranking `[poi_net_utility, pier_gain, matrix_retention,
lower_intervention_energy]`, floor 0.5, `forbidden_inputs=[confirm,phase_b,seame,transfer]`.
Top-2 per (family, model) recorded in `top2_by_family`. **All six frozen final settings are
members of their family's Phase-A top-2** (verified). The `FINAL_REPORT` "candidates" line is a
positive-on-both-corpora narrative filter, not the selection.

## 10. Qwen rescue audit — PASS
`QWEN_RESCUE_DECISION = NOT_TRIGGERED`. Rule = fire iff no Qwen family has a selected setting with
positive Search utility ≥ floor. Qwen `add_unique/encoder` had Search `poi_net_utility=+1` (both
top-2 members) → rule-correct NOT_TRIGGERED. (That tiny Search signal did **not** replicate in
Phase-B — a visible winner's-curse, not a rule error.)

## 11. Confirmation eligibility — PASS (method-specific, explicit)
`CONFIRMATION_ELIGIBILITY_SUMMARY` (`outcome_artifacts_loaded=false`):
- **CS**: acoustic (encoder) eligible **128**, transcript (decoder) eligible **280**,
  `structurally_ineligible_acoustic=152`. Encoder N=128; decoder N up to 274–280 after the
  per-model baseline-causal-position gate; **COMMON=128**.
- **ASCEND**: 200/200 both sides; COMMON=200.
- **SEAME-man/sge**: acoustic 50, transcript 49 (1 transcript-ineligible each), FULLY_ELIGIBLE 49.
Method-specific eligibility is handled explicitly (no silent row drops); no confirmation sample
was replaced after Phase A.

## 12. Phase-B completeness — PASS
Both decode modes present (greedy 6,825; official_standard 6,257 rows overall). Frozen top-2/family
evaluated on the held-out eligible confirmation sets. **Qwen `official_standard ≡ greedy`** is
proven per row (`provenance.decode_equivalence = "official_standard == greedy"` +
`equivalence_hash`; manifest `qwen_official_equivalence` hash) — no invented beam, no duplicate
physical compute claimed.

## 13. Confirmation metrics — PASS
Rows carry MER/PIER/EN-WER/matrix-CER-gain, corrections, corruptions,
correction_minus_corruption, outside/retention (embedded+matrix), transcript-change, plus
`hidden_perturbation`/`mean_edit_norm`; `target_logit_diagnostics` present-but-null in the
accepted physical schema. Every table row carries N total, N eligible, eligibility rate,
`eligibility_scope`.

## 14. Common-subset comparison — PASS
Rows emit `eligibility_scope ∈ {family_specific, common}`; cross-family comparisons
(encoder-vs-decoder, Add-Unique-vs-Conditioning-CS) have explicit common-eligible rows. The
report instructs use of common rows, not pooled denominators.

## 15. Final-setting selection — PASS
Exactly one setting per (family, model): Whisper {AU-enc L20/ρ1.0, AU-dec L24/ρ0.5, Cond-CS-dec
L0/ρ0.5}; Qwen {AU-enc L15/ρ2.0, AU-dec L13/ρ0.5, Cond-CS-dec L0/ρ0.5}. `PHASE_B_FINAL_SETTINGS`
`seame_outcomes_loaded=false`, `status=FROZEN_BEFORE_PHASE_C`, `source_confirm_sha256` present.
- **Note:** the Whisper encoder final (L20) maximizes pooled held-out utility (CS −1 / ASCEND
  +17) but is **CS-negative**; the encoder family's positive-on-both evidence rests on the *other*
  top-2 member (L4: CS +1 / ASCEND +9). Recorded as a cross-corpus-consistency caveat (§ claims).

## 16. SEAME transfer audit — PASS
`PHASE_C_MANIFEST` `status=FROZEN_BEFORE_TRANSFER_EXECUTION`, `seame_outcomes_loaded=false`,
`seame_eligibility` created before execution, `decode_modes=[greedy, official_standard]`,
`fixed_direction_artifact=null`, references `phase_b_final_settings` hash. Only the frozen final
settings were evaluated; SEAME is transfer-evidence-only and was unlocked after Phase-B freeze.

## 17. Reuse / provenance — PASS
`REUSE_LEDGER`: 16 baselines `REUSE_EXACT` (each with source_hash + validation_hash,
`validation_errors=[]`) + 1 `PARTIAL_REQUIRES_VALIDATION` (Whisper CS greedy encoder Add-Unique
L0 direction cache — an oracle-TT artifact, validated). **No corpus-level fixed-steering row was
reused as per-sample oracle-TT.** Reuse equality is hash-based, not filename-based.

## 18. Slurm / job audit — PASS
`runtime/PHASE_BC_JOB_AUDIT` (+ `migration/SLURM_JOB_AUDIT.md`): `unrelated_jobs_cancelled=false`;
superseded/failed shards cancelled only after preserving completed artifacts; duplicate Qwen CS
decoder coverage deduped (valid atomic rows retained); a resubmit avoided overlapping shard
coverage. No required shard missing (Phase-A validation missing=0). The earlier large-A6 atlas
jobs (53354/53355) remain cancelled; `results/basis_a6_expanded/` preserved.

## 19. Runtime — measured
- Whisper GPU-h (Phase A, incl. failed pre-fix): **1.495**; Qwen: **1.517**.
- Phase-B all attempts: **2.548**; Phase-C all attempts: **0.297**.
- **Total ≈ 5.86 GPU-h** (vs ~44 GPU-h conservative preflight → ~7.5× lower, from one-pass
  all-layer analysis + rho-cache reuse + method-specific eligibility shrinking CS confirm to 128).
- Wall clock (first search start → last transfer end, 2026-09-21): **13:02:37 → 17:14:42 ≈
  4h12m**, including all repair/resubmit cycles and firewall-imposed sequential phases.
- GPU-h avoided by reuse: the 16 baseline decodes (not recomputed) + Whisper enc AU L0 cache.

---

## Whisper oracle upper bound (held-out Phase-B)
- **add_unique/encoder — MIXED.** Positive-on-both candidate exists (L4: CS +1 / ASCEND +9), but
  the frozen final (L20) is CS-negative / ASCEND-positive; retention ≥ 0.997. Cross-corpus
  inconsistent.
- **add_unique/decoder — CONFIRMED_POSITIVE (small).** L24 held-out CS utility +13, ASCEND +6,
  corrections > corruptions, matrix retention ≈ 0.997–0.999. Small absolute effect (PIER gain
  ≈ 0.006–0.014) but positive on both corpora.
- **conditioning_cs/decoder — MIXED.** Frozen L0 positive on both (CS +8 / ASCEND +15), but the
  family's other top-2 candidate (L11/ρ2.0) is damage-dominated (CS utility −22, retention 0.986).
- **Overall — MODERATE positive oracle mechanism; gate PROCEED_TO_NON_ORACLE** (driven by decoder
  Add-Unique and the chosen Conditioning-CS setting; effects small but held-out and positive on
  both corpora).

## Qwen oracle upper bound (held-out Phase-B)
- **add_unique/encoder — NO_CLEAR_SIGNAL / mild DAMAGE** (utility −3/−4).
- **add_unique/decoder — NO_CLEAR_SIGNAL** (utility +1/−2/0; retention ≈ 1.0).
- **conditioning_cs/decoder — NO_CLEAR_SIGNAL** (utility −4/−2/0).
- **Overall — ORACLE_MECHANISM_NOT_ESTABLISHED.** No family is positive on both held-out corpora.

## Qwen causal diagnosis
Hidden perturbation is **real** (`mean_edit_norm ≈ 10`) yet decode is essentially unchanged
(matrix retention ≈ 1.0, near-zero corrections). `target_logit_diagnostics` is null in the
accepted physical schema, so the run **cannot distinguish** LOGIT_RESPONSIVE_BUT_NOT_DECODE from
HIDDEN_RESPONSIVE_BUT_LOGIT_WEAK. Classification: **NO_CLEAR_SIGNAL / DAMAGE_DOMINATED, cause
UNRESOLVED.** The screen swept ρ∈{0.5,1,2,4} and many layers (encoder + decoder) without signal,
which weakly argues against "insufficient ρ / wrong single layer" and toward an
intervention-site / causal-pathway or large-decision-margin explanation — but with logit-movement
data absent, no single cause is forced.

## Search → Confirmation generalization
Whisper decoder Add-Unique (L24) and Conditioning-CS (L0) search winners **replicated** with
positive held-out utility. Qwen's Search encoder signal (+1) **failed to replicate** in
confirmation — a visible winner's-curse from the 40-sample search. Selected ρ/layer trends for
Whisper decoder generalized; Qwen did not. Search and confirmation metrics are kept separate
throughout.

## Cross-dataset transfer
Within-domain oracle upper bound: **ASCEND is consistently more responsive than CS** for Whisper
(e.g. encoder L20 ASCEND +17 vs CS −1; decoder/Cond-CS ASCEND positive). CS within-domain effects
are small and, for encoder, inconsistent. SEAME is transfer-evidence-only (read post-freeze); it
must not be read as broad generalization. No broad cross-dataset generalization is claimed.

## Claim strength
- **STRONG:** none (no held-out effect is large; all corrective effects are small).
- **MODERATE:** Whisper decoder Add-Unique (L24) positive-on-both held-out, small, high retention;
  Whisper Conditioning-CS (L0) positive-on-both held-out, small.
- **EXPLORATORY:** Whisper encoder Add-Unique (cross-corpus inconsistent); SEAME transfer; Qwen
  rescue-not-triggered diagnostic; Qwen hidden-perturbation-without-decode-response observation.
- **UNSUPPORTED:** any claim of a Qwen oracle-steerable mechanism; any strong within-CS correction
  claim; any dose/site attribution of the Qwen null.

## Next-stage gate
- **Whisper: PROCEED_TO_NON_ORACLE** — held-out positive-on-both decoder mechanism, high retention.
- **Qwen: ORACLE_MECHANISM_NOT_ESTABLISHED** — no held-out signal; the oracle upper bound itself
  is null. (Gate is intentionally per-model; a weak Qwen result is acceptable.)

## Recommended non-oracle target
- **Whisper:** `add_unique/decoder L24 ρ0.5` (primary non-oracle learning target); `conditioning_cs/decoder L0 ρ0.5` (secondary). Encoder Add-Unique is not recommended (cross-corpus inconsistent).
- **Qwen:** none. Do not proceed to non-oracle steering. First diagnose the causal pathway by adding
  target-logit-movement instrumentation to determine whether the real hidden perturbation reaches
  the output distribution at all.

## Freeze artifacts
`FINAL_AUDIT.md` (this) + `FINAL_FREEZE.json` (manifest/setting/result/artifact hashes, model
revisions, eligibility counts, runtime, gates, audited commit). Scientific result files are
unmodified.

## Exact remaining blocker
**NONE.** A6-OTT is frozen.
