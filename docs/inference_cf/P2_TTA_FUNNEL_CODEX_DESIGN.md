# Implementation handoff — P2-TTA-FUNNEL

Design only: no new runner or outcome was executed in this freeze. Scientific contract is `P2_TTA_FUNNEL_SPEC.md`, machine contract `configs/inference_cf/p2_tta_funnel.json`, immutable MAP panel/algorithm and TTA1 inheritance documents. Historical TTA0 INVALID and all its bytes stay unchanged. Core v6 authority stays unchanged; this is a separate exposed-development ticket.

## Reuse and implementation changes

- `src/csasr/inference_cf/episodic_tta.py`: reuse LNGuard, fp32-master functional_call casts, allowed_ids/valid_mask, position_terms, teacher_logits, adapt/reset mechanics and forced_decode. Extend with explicit student condition and A3 objective; preserve A1/A2 exact regression. Compute detached AUTO teacher probabilities on the identical y_A prefix once. Do not reuse a FORCED teacher as AUTO merely because historical decoded transcript agrees.
- `src/csasr/models/whisper.py` and `experiments/inference_cf_p2seq.py`: preserve historical greedy baseline paths/model/audio. Actual AUTO prompt recovery uses installed transformers4.57.6 WhisperGenerationMixin.detect_language and historical generation_config on theta0 encoder output; teacher forced forward then uses recovered language token. No assumption AUTO=EN. Recover teacher condition before any parameter update; verify source/version and generation-config hashes in preaudit.
- `experiments/inference_cf_p2tta0.py`: reuse teacher seal, decoder-LN parameter enumeration, detached encoder, saved masters, logging and zero-steering final decode. Its historical run/config remains immutable. New funnel runner should implement conditional stages explicitly; R must not call adapt/optimizer/model generation/backward. MAP runs two objectives in one allocation; TTA1 resolves one from the selection seal.
- `experiments/inference_cf_p2tta0_analyze.py`: reuse canonical metric/count and retention primitives, but its current analyze entry reads references even when integrity fails. **Do not invoke that entry before the new R evaluation gate**. New guarded evaluation wrapper copies frozen original decision semantics, opens references only after gate and byte verification; no alteration to old report or thresholds.
- `experiments/inference_cf_p2tta0_audit.py`: reuse independent arithmetic/reconstruction ideas. New auditor must not import primary objective/analysis/decision code. Independent A3 full-vocabulary KL plus float64 identity, complete-master gradients with frozen2% numerical tolerance; fixed masks/parameters verified separately. For R use existing sealed live checks and diagnostic only, not a new adaptation run.
- Canonical evaluator in `src/csasr/evaluation/{canonical,retention,mer,pier}.py`; P2SEQ outside-POI lexical harm and original TTA0 bootstrap/count rules. Primary and auditor may share locked canonical primitives, but not decisions/bootstrap aggregation implementation.
- Existing `csasr.utils.provenance` / `csasr.lss.specfreeze` manifest mechanisms and Slurm conventions. No gate/direction/core-method modifications.

Add minimal `experiments/inference_cf_p2tta_funnel{,_analyze,_audit}.py`, `slurm/inference_cf_p2tta_funnel.sbatch`, and `tests/test_inference_cf_p2tta_funnel.py` during Claude implementation only. Extend episodic_tta.py only for A3/condition diagnostics; do not duplicate model loaders/evaluation. Use stage directories `results/inference_cf/p2tta_funnel/tta0_r`, `/map/run1`, `/tta1/run1`. Later stage reports `P2_TTA0_R_REPORT.md`, `P2_TTA_MAP_REPORT.md`, `P2_TTA1_REPORT.md`; none created now.

## Execution sequence and seals

1. Verify all master source/sealed hashes. Contract-level tests first. Audit existing run1 numerics with only tolerance override, preserve original invalid record. New R independent gate **PASS_TO_P2_TTA0_R_EVALUATION** must be pushed before reference evaluation. Zero new adaptation/model inference.
2. Evaluate sealed run1 using original TTA0 rules. Independent **P2_TTA0_R_AUDIT: PASS**; push R report. Select→push immutable selection, skip MAP and go TTA1. None→MAP. Still-invalid→STOP all.
3. Only conditional MAP: independently replay frozen24 selection; no new IDs. Seal teachers/config and resolved manifest before outcome; independent **PASS_TO_P2_TTA_MAP**, push. One job≤60min. Output seals and **P2_TTA_MAP_AUDIT: PASS** precede diagnosis/branch. None/invalid stops; selected→push exact selection before TTA1.
4. Only conditional TTA1: verify inheritance/panel100, audit all planned reuse, resolve one objective and episode origins; independent **PASS_TO_P2_TTA1**, push. One job≤3h. Seal outputs; **P2_TTA1_AUDIT: PASS**, report and STOP regardless of verdict.
5. Every reviewed edit: checks→diff review→commit→push origin HEAD:cs-asr-steer-inf. No force/main merge/PR. Budget/partial audit failure cannot authorize another job or altered formulation.

## Focused tests required before future execution

- Original config's complete scientific decision section matches master copy; R alters only numerical relative-gradient tolerance. All run1/output/checkpoint/baseline hashes unchanged. R call graph forbids adaptation, backward, pseudo generation and optimizer steps; references unavailable before gate.
- Original A1/A2 losses, allowed/suppressed vocabulary, content mask, y_B/y_A cleaning/prefix/EOS/cap, forced decode and optimizer/reset regressions.
- Exact194 decoder-LN-only trainables; encoder/non-LN grads absent; fp32 master/bf16 differentiable casts; exactly2steps; reset bitwise; fresh optimizer/caches per episode.
- MAP selector deterministic24 on real theta0 input hashes; D/A exact text equality; diversity round-robin and least-represented fill; LOW_DISAGREEMENT and insufficient-A fixtures; no evaluator fields accessed. Panel24 byte identity and fixed100 identity.
- A3 independent full-vocabulary KL atT1, same content prefix/allowed set, teacher stop-gradient; no top-k; AUTO native language prompt (including AUTO=ZH zero-gap fixture); no forced-EN substitute; loss/mask formula float64 equivalence/repeat mixed precision; finite gradients and no teacher/encoder grad.
- Diagnostics L0/L1/L2/D_cond, token-weighting and zero-gap guards; thresholds inclusive; exact primary diagnosis precedence and secondary safety flags. Ties1e-12 and final A3/A2 tie rules separately.
- TTA1 inheritance rejects changed objective/LR/steps/prompt/mask/subset; only selected objective; reuse determined by provenance, not outcome; all100 complete. Objective labels, stage branch guards, point safety thresholds, severe truncations, bootstrap deterministic. No steering/D2/continual/TTA beyond frozen branch invoked.

Independent post-audit reproduces loss/update/reset mechanics, sealed IDs/teachers/settings, canonical counts/metrics, paired bootstrap, labels/ties and next branch. No scientific conclusion on failed audit. No new giant framework: extend existing focused components.

Historical source hashes in master config anchor the inspected starting code. If episodic_tta.py is extended, verify those historical bytes through git at starting_commit, run original regression tests, and freeze new implementation hashes in the resolved manifest. This does not permit changing immutable scientific settings/results/panels or tolerances.
