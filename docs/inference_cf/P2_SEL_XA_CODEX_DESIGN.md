# P2-SEL-XA — codebase-aware implementation handoff

Freeze base `5186b5caba06ae1bd4ede7a340e10088a0fd541c`, clean local==origin/cs-asr-steer-inf;
no intentional local-only changes or queued jobs. Spec/config are the experimental authority for
this bounded stage; METHOD_CONTRACT remains the core authority. No runner/scientific job implemented
or run in this design session. Parent E/T stops and reports unchanged.

## Actual implementation findings and reuse

| Requirement | Existing path / action |
|---|---|
| q/u/r | src/csasr/lss/sites.py:DecoderPostCrossAttnRecorder captures LN pre-hook q and encoder_attn output[0] u; addition in model dtype, detached float32 records. DG-02 assert_site_reconstruction has dtype-aware max(1e-3,4*eps) tolerance. |
| J/g_J | src/csasr/inference_cf/readout.py:objective, ZeroProbe, readout_direction, clone_scratch. Raw out['gradient'] is separate from tangent/unit direction. Frozen generate suppression/partition/float32 autograd. Never import p2rj SiteGradientProbe/evaluator for runtime. |
| Existing gradients | p2dir.py writes t{t}_g_readout to exp1_run1/rows/{i:03d}.npz; directions_sealed.json hashes NPZ and NONE logits. All180 float32 length1280 raw vectors verified present/hash-identical by inventory only; no XA dot products computed during design. Mandatory reuse in XA0. |
| B replay | experiments/inference_cf_p2r.py:DiagBranch / inference_cf_cached.py:Branch.step. Replay only prompt+baseline tokens preceding t; record exact current last-query source. |
| Population/gates | p2rj/positions.json; p2dir/construction_population.json; p2sel.join_gates/load_p2dir_rows; P2-R run3 E/R_B/g. Same historical42/18/55/5/60. |
| E_tok | p2sel_t/t0_run1 compact rows plus manifest/audit; descriptive only. No native_lid or windows rerun. |
| XA1 old/new | P2-SEL C0/C2 OLD gated D2, sealed D2/NONE; p2sel_e.e1_utterance + p2sel.gated_hook. New gain only, no OLD-direction C1 or BROAD. |
| CPU evaluator/stat | p2dir_analyze.logit_metrics/draw_weights/boot_stat; p2sel_e_analyze group/paired patterns; p2sel_t auditor independence pattern. No inherited E/T branch code drives XA decisions. |
| Exact energy | independent p2sel_audit bf16 hook emulation; do not compare tiny bf16 edits to ideal float64 norm only. |
| Decode/panel | cached B/E/S primitives, live readout provider, canonical metrics; existing P2_SEL_MINI_PANEL.json100IDs/20dlg. No conditional mini runner exists yet; add only required orchestration, no new panel. |
| Provenance | csasr.utils.provenance / csasr.lss.specfreeze and existing inference_cf manifests; parent/file/code/config/model hashes, clean pushed sources, runtime/status and audit links. |

## Minimal additive implementation

After freeze, add `experiments/inference_cf_p2sel_xa.py` (manifest/xa0/conditional xa1, later conditional
xa2), `_analyze.py`, `_audit.py`, `slurm/inference_cf_p2sel_xa.sbatch`, and
`tests/test_inference_cf_p2sel_xa.py`. A small reference-free source-compatibility helper may live in
`src/csasr/inference_cf/source_compatibility.py` for both diagnostic and live decode. It receives only
raw g/u, not labels/evaluator signals. No changes required to readout.py/core_r2/sites/hooks or historical
runner/formula/config/report files; reuse public primitives and narrowly extend orchestration.

XA0 per utterance: load verified raw gradients/NONE/direction sources, encode once, replay B. For selected
query clone pre-step cache/encoder first (no live mutation), then clean B.step under
DecoderPostCrossAttnRecorder(...,keep_last_only=True), close recorder and compare r/logits bitwise to
seal. Extract q/u/r. One isolated clone forward with current-query encoder_attn output[0] scaled exactly
0.95; all other output tuple fields unchanged. Capture perturbed r; hooks removed on any exception.
No encoder re-encode, persistent token sampling, LID, evaluator gradient or backward. CPU compute J
with frozen objective and seal's raw gradient S/C/F; no other source/head decomposition. Commit compact
rows and vector/logit hashes; local binary arrays retained with manifest checksums.

Do not wrap readout_direction in a live recorder context: its assert_no_site_hooks would correctly reject
those hooks. XA0 never needs a new readout call. XA2 obtains the live raw gradient and D2 from one
readout_direction call on pre-step B cache before opening the clean-step recorder, then executes B.step
and closes that recorder. Verify scratch-vs-clean site/logits identity, compute S from returned raw
 gradient and clean u, and run the existing S/D2 hook with selected gain. This adds no second backward.
Direction remains the existing tangent-normalized result, while F uses raw gradient. Existing tiny/
nonfinite D2 no-direction fallback remains no-edit; invalid source feature fails manifest/audit, never
uses another factor. Do not use teacher-forced q/u/F after a live prefix diverges.

The fixed .95 diagnostic is not a scientific actuator or an XA1 arm. It is absent in runtime XA1/XA2.
Nonlinear/quantized finite-difference failure invalidates this bounded first-order interpretation;
no lambda/tolerance/precision retry. F is fixed clip(S,0,1), even if cosine looks more discriminative.
Historical EN_FN remain gated off because E=0; recall-only is a stop, not an authorization to remove E.

## Focused tests and independent audit

- Ordered180 uid/t/query identity, group counts and exact seals/mini-panel hashes; no refill/regroup.
- Recorder returns direct q/u, native-dtype sum bitwise r; float64 reconstruction obeys frozen dtype
  tolerance. Capture before FFN; eval/dropout and forced-prefix/query position checked.
- J/processed suppression and raw-gradient identity unchanged; no tangent/unit vector substituted.
  Reuse existing D2 objective/autograd/no-grad/cache/future-prefix tests.
- Hand-derived raw dot/norm/C and clip boundaries (negative/zero/0.1/0.7/1/large S); zero norm gives
  exact0; nonfinite fails; C denominator eps and signs. No learned scale/threshold/alternate factor.
- Lambda1 exact identity and .95 scales only target query's encoder_attn output, q unchanged; clone
  has no persistent pointers/cache mutation. Synthetic smooth J sign/magnitude check; bf16 realized
  delta captured/recomputed exactly, material-subset/sign/ratio/RMS boundaries deterministic.
- No retained parameter gradients, hooks/graph leaks, labels/reference/evaluator/future inputs.
- One compatibility formula only; exact XA0/XA1 precedence, support/dialogue/CI/count boundaries,
  missing-draw INVALID, recall zero-E stop; no cosine fallback or sweep.
- Zero new gate bitwise NONE, old D2/R_B/E/site/alpha regression and NormPreserve exact bf16 energy;
  independent auditor may reuse metric primitives but never import primary decision code.
- Existing100 panel and live-prefix construction for decode; no dataset role expansion.

Independent XA0 audit recomputes J from raw saved logits, q/u/r and g source hashes, S/C/F, fixed
finite difference/subset, shared bootstrap, all booleans and label. XA1 independently recomputes exact
bf16 edit, gate, evaluator deltas/corruption/CIs/verdict; XA2 canonical metrics/live-feature lineage.
Manifest/source identity checks precede outcomes. No conclusion without required stage PASS.

## Exact Claude execution order

1. Implement additive primitive/runner/analysis/independent audit/tests; check→diff→commit→push.
2. CPU `PASS_TO_P2_SEL_XA_XA0`: verify parents terminal/audited, all180 sealed raw gradients/NONE/key/
   source/provider metadata, panel/firewall and no XA outcomes. Commit/push pre-run audit/manifest.
3. One XA0 sbatch job, then fixed CPU decision and independent `P2_SEL_XA_AUDIT: PASS (XA0)`.
   Push auditable rows/statistics/report. INVALID/recall-only/not-discriminative all STOP.
4. Supported only: immutable xa0_selection.json with sole factor, hashes/predicates/audit; commit/push
   BEFORE XA1 outcome. Pre-XA1 lineage audit; one NEW pulse job reusing exact OLD/NONE.
5. Independent XA1 audit PASS and frozen label; every label except GATE_SUPPORTED STOP.
6. Supported only: verify exact100 panel, one conditional B0/OLD/NEW mini job; independent XA2 audit
   and descriptive report. Always STOP; no300, P3, new data, extra direction/gate/lambda/precision.
7. Each reviewed edit checks→diff→commit→push; preserve failures/partial runs, no automatic extra
   job after time exhaustion. Final clean tree and local==authoritative remote branch.

No scientific choices are delegated to implementation: numeric criteria, finite-difference point,
material subset, F, composition, C disagreement policy, thresholds, budgets and stops are frozen.
