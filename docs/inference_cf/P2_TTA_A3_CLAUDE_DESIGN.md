# P2-TTA-A3 — implementation design (Claude, pre-outcome)

Scientific authority is `P2_TTA_A3_SPEC.md` together with `configs/inference_cf/p2_tta_a3.json`. This file describes
the minimal implementation, which reuses the audited A2/TTA1 machinery. Historical modules are not modified:
`episodic_tta.py`, the TTA0 runner/analysis/auditor and the funnel code stay byte-identical, so their past audits
remain reproducible.

## Reuse

- `csasr.inference_cf.episodic_tta`, reused unchanged: `decoder_ln_names`, `LNGuard` (theta0 snapshot, restore and
  verify, fresh fp32 masters), `teacher_logits` (bf16 `functional_call`, `use_cache=False`, query 3 + t), `allowed_ids`,
  `valid_mask`, `clean_teacher`, `OPTIM`/`STEPS` (AdamW lr 1e-3, 2 steps), `_l2`, `forced_decode` (cached Branch,
  processed argmax), `levenshtein`, `compare`, `severe_truncation`, `tensor_bytes_hash`.
- `experiments.inference_cf_p2tta0`: audio fingerprints and the `_summ_common` shape.
- The TTA1 resolution of A2 outputs: TTA1 run1 rows for new IDs, and TTA0-R run1 rows for reused ancestors, from
  `results/inference_cf/p2tta_funnel/tta1/plan_sealed.json` with hash checks.
- Canonical evaluator primitives (`csasr.evaluation.*`) and the P2-SEQ outside-POI lexical harm / `utt_counts` /
  `draws`.

## New code

- `src/csasr/inference_cf/soft_auto_tta.py`:
  - `auto_prompt(lang_id)`;
  - `teacher_distribution(model, enc, cA, y, sup, beg)`: float32 detached log q per step over the allowed IDs;
  - `kl_terms(student_logits, logq_list, ...)`: per-position forward KL;
  - `adapt_a3(...)`: a mirror of `episodic_tta.adapt` with the identical optimizer/master/step mechanics and the KL
    loss. It applies the identical-condition / empty-valid rule, connecting a scalar zero to the masters.
  - It logs D_cond per step, gradient norms, master/effective deltas, and student entropy/P_E/P_M/EOS on the y_A path.
- `experiments/inference_cf_p2tta_a3.py`: `panel` / `prepare` / `manifest` / `run` / `seal`.
- `experiments/inference_cf_p2tta_a3_analyze.py`: mechanistic diagnostics, canonical metrics, frozen precedence;
  references are loaded only behind the committed output seal.
- `experiments/inference_cf_p2tta_a3_audit.py`: independent `prerun` → `PASS_TO_P2_TTA_A3`; the live
  `live_kl_check` (independent full-vocabulary masked formula) used by the runner on row 0; and `post` →
  `P2_TTA_A3_AUDIT: PASS`. It does not import the analysis or `soft_auto_tta`, and recomputes the panel, D/A, D_cond,
  distances, movement, metrics, safety and label.
- `slurm/inference_cf_p2tta_a3.sbatch` and `tests/test_inference_cf_p2tta_a3.py`.

## Run order (one allocation)

Load bf16 Whisper with the decoder LN guard. Then, for each of the 24 rows in panel order:

1. compute the historical-feature encoder once (detached, cloned outside inference mode);
2. theta0 forced decode, which must equal S0;
3. `detect_language(input_features=...)` → cA;
4. theta0 AUTO replay decode under cA, which must equal the historical AUTO text;
5. row 0 only: the independent live KL check;
6. the A3 episode (teacher computed once before updates, 2 steps, D_cond0/1/2);
7. materialize the effective bf16 values, then forced-ZH final decode with a fresh KV cache;
8. in a `finally` block, restore theta0 bitwise and verify.

Save rows and final masters (tiny npz). Teacher logits are never saved.

## Gates and commits

1. Freeze spec/design/config/panel (with the panel builder) → push.
2. Implementation, tests and CPU `prepare` (plan seal) → `PASS_TO_P2_TTA_A3` → push.
3. Manifest pushed as a fast-forward child commit before the job → one sbatch.
4. Output seal → push.
5. Evaluation → independent post-audit → report/docs → push → STOP.
