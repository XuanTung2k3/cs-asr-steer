# P2-PATH0 — implementation handoff

Design-only freeze: no runner implementation or scientific GPU outcome in this session. Normative spec/config/sites files freeze the experiment. Historical A2/A3/A4 modules/configs/reports/results must remain unchanged. This is a separate exposed-development diagnostic, not a core v6 method change.

## Actual implementation map and reuse

- `src/csasr/inference_cf/script_safe_tta.py::adapt_a4`: unchanged reconstruction, q_SAFE selection and exact no-op rules. Calls `soft_auto_tta.teacher_distribution/allowed_log_probs/kl_terms` and `episodic_tta.teacher_logits`; actual optimizer/steps constants are inherited. Safe teacher uses theta0 AUTO only at canonical embedded-script targets and theta0 forced elsewhere.
- `src/csasr/inference_cf/episodic_tta.py`: `decoder_ln_names`, `LNGuard`, `tensor_bytes_hash`, ordinary `forced_decode`, `levenshtein`. LNGuard materializes bf16 effective tensors and restores exact theta0. The194 names/248320 scalars are frozen in A4 config and copied verbatim into PATH0.
- `experiments/inference_cf_p2tta_a4.py::cmd_run`: reuse exact audio fingerprint/full SHA, encoder inference/clone, model/env/partition/suppression checks, theta0 integrity decode, `detect_language(input_features)` matching sealed language,2-step adapt, checkpoint comparison and final reset pattern. Do not invoke its24-row runner for PATH0.
- `experiments/inference_cf_cached.py::Branch.step`: fresh isolated cache, explicit cache_position, prompt first then one consumed token per call, passive post-cross-attention recorder at16, attention=True. `core_p1.processed_argmax` and cached `_processed` implement suppression/begin suppression. New clamp path must preserve these calls and CPU-float32 argmax behavior exactly.
- `results/inference_cf/p2tta_a4/plan_sealed.json`: sealed y_B/y_A, their termination, teacher masks/classes, audio/model hashes and A3 lang IDs. `output_seal.json`: A4 output/checkpoint byte hashes. `run1/rows/{00..11}.json`: D A4 outputs (parent A3 indices frozen in sites). Their adjacent `_final_masters_fp32.npz` files are integrity comparators, not substitutes for reconstruction.
- Parent `docs/inference_cf/P2_TTA_A3_PANEL.json`: exact D rows/order and historical FORCED/AUTO paths. PATH0 site artifact stores all three token sources; never use text divergence. The fixed100 mini-panel is lineage only, no100-row execution.
- `experiments/inference_cf_p2tta_a4_analyze.py` / `_audit.py`, P2SEQ and canonical evaluator: reuse existing per-row canonical errors/counts and retention. PATH0 primary analysis is reference-free suffix ED; reference scoring must be a separate guarded command after primary seal. Do not run the historical A4 evaluator to derive sites.
- Existing provenance/atomic JSON/digest utilities and Slurm templates. New manifest records config/sites/source/model/environment/git/audio/checkpoint hashes, resolved12 IDs, budget, status and zero reference access.

## Files to add later, without historical changes

`src/csasr/inference_cf/path_decode.py`: narrow clamp decoder matching forced_decode; no objective/hook changes. Pure decision-stream conversion/site validation/suffix metrics may live here or separate pure helper. No reference imports.

`experiments/inference_cf_p2path0.py`: CPU preparation/preaudit/manifest, single GPU two-phase run, immutable primary seal. `experiments/inference_cf_p2path0_analyze.py`: reference-free primary analysis plus separately guarded secondary evaluator. `experiments/inference_cf_p2path0_audit.py`: independent site/ED/decision/flag implementation, no import of primary site/decision functions. `slurm/inference_cf_p2path0.sbatch`; `tests/test_inference_cf_p2path0.py`.

Output directory `results/inference_cf/p2path0/run1`; prep/gates/seals at `results/inference_cf/p2path0/`. Expected later `docs/inference_cf/P2_PATH0_REPORT.md` only after audit. No such report now.

## Exact execution order

1. Validate historical source/config/panel/seal/row/checkpoint hashes; independently reconstruct C/U/sites and donor spans from raw content arrays plus termination. Check source tokenizer controls/suppression, common prefixes and12 IDs. All design files must already be pushed. New runtime code/manifest must be hash-frozen and pushed too.
2. Focused tests and independent **PASS_TO_P2_PATH0**, commit/push. Prepare resolved run manifest and commit/push. Submit one30-minute sbatch allocation; target<10 minutes.
3. Phase1 all12: original theta0 model/encoder, theta0 forced replay and branch scalars; recover same sealed AUTO condition; call unchanged adapt_a4 once; exact effective-state/checkpoint comparison; ordinary FREE-A4 reproduction tokens/termination; capture adapted branch scalars; restore and verify. Retain detached encoders and CPU effective states. Only24 optimizer updates total. Independent original live first-row formula check adds at most1 backward without update. **Do not begin a clamp until all12 pass.**
4. Phase2 each row: materialize its reconstructed A4 state; run fresh-cache SELF-1 first, verify complete FREE identity; scientific1/3 donor clamps; restore in finally. Assert all non-LN tensors unchanged, no adaptation/update inside phase2, no state/cache carried between arms. Save compact output tokens/termination and trace: each forced decision, donor token, suppression validity, cache length/positions, release index, prefix identity. Do not save vocabulary logits.
5. Seal complete48 adapted decodes (FREE/SELF/two clamps per row), branch scalars, state/reset/caches, suffix distances and primary label inputs/reference-free verdict. Commit/push **before reference access**. Independent primary audit may verify seal first; failed primary audit blocks references and yields INVALID.
6. Secondary evaluation only for12 exposed rows, with predeclared ASR flag; independent full **P2_PATH0_AUDIT: PASS**, commit/push report and STOP. No next objective/search/deployment code automatically follows.

## Focused test contract

- Exact12 D IDs/order,5/7 C/U, original arrays/termination and all byte hashes. Separate pure site reconstruction agrees with frozen JSON.
- Zero-based content indexing excludes prompt. Logical EOS only for `terminated=eos`; real EOS strict-prefix site; cap-prefix undefined site rejected; no fabricated EOS/control stripping; unavailable three-token tail shortened deterministically; forced EOS stops immediately.
- Common-prefix hash compact JSON convention. Donor1/3, effective length/release indices, requested-vs-effective lengths exactly match artifact.
- New no-clamp decoder exactly matches historical forced_decode; greedy/suppression/begin-at-step0/cap200 and prompt unchanged; same token-by-token cache positions. SELF identical under toy deterministic model and real mandatory runtime checks. Independent caches across arms; no batched prefill or shared mutable KV.
- Forced token consumed before next logits; forcing suppressed token rejected; pre-site greedy equality required. Zero reference/label/teacher quality access constructs clamps. No optimizer or backward in clamp phase.
- Reconstruction invokes unchanged A4 twice-updated194-LN path; theta0/A4 exact token/termination and checkpoint bf16 identity; all12 barrier prevents partial-valid scientific run; resets and non-LN identity tested.
- Suffix slicing at k+effectiveL on all source/intervention streams. Forced-span-only improvement cannot pass. DIRECT_ONLY excluded, negative ratios retained, all eligible failures included in pooled ratio. Induction success requires downstream difference from B0.
- Current eligible counts R1/R3=4/4, I1/I3=6/5; successes3 each plus2 dialogues; small-denominator false, exact inclusive50%/30% thresholds, all label precedence fixtures (including mixed1/3 asymmetry).
- Secondary alignment flag gated after primary seal, same-L causal pass and integer ZH/POI counts; reference results cannot mutate primary verdict. Independent auditor imports no primary decision logic. Budget/missing/nonfinite failures stop without refill/retry.

Historical source hashes anchor starting_commit; new additive runtime sources are frozen separately in the resolved manifest. Existing A4 adaptation/model/cached primitives must remain byte-identical. A genuine compatibility contradiction blocks execution; do not silently repair A4 or change scientific settings.
