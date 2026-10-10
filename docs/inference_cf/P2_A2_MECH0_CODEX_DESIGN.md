# P2-A2-MECH0 — implementation-ready mechanism handoff

Read SPEC/config/fingerprinted PANEL first. A2 stays exact; PATH controller line is closed.
Design session creates no scientific runner/outcome/report. Implementation is later Claude work,
with independent PASS_TO_P2_A2_MECH0 before its single job. This is already-exposed development,
not the core v6 method or a fresh confirmation role.

## Reuse and mechanical compatibility

- `experiments/inference_cf_p2path5.py`: source model/audio/teacher processing, exact original A2 episode,
  canonical live theta0 forced-ZH decoding, LN materialization/hash/reset and output manifest/seal.
  Reuse these primitives; do not copy its G1A/controller phases or opportunity gate.
- `src/csasr/inference_cf/episodic_tta.py`: decoder_ln_names, LNGuard, fresh masters, A2 objective_loss/adapt,
  teacher_logits, valid_mask, allowed_ids, forced_decode, tensor_bytes_hash, severe_truncation.
  Two updates and three losses are already implemented. The missing capability is detached STEP1 snapshot.
- `experiments/inference_cf_p2tta0.py::run_objective`: authoritative hard-AUTO NLL episode and reset.
  Optional observer forwarding is mechanical instrumentation only; A2 loss/optimization unchanged.
- PATH5 plan/seal/run rows: authoritative FULL300 order and canonical B0/A2 (not P2-A historical B0),
  all300 effective A2 state hashes. Final-master archives exist for FIXED100; NEW200 has effective
  state hashes but no sealed final-master archive. Compare exactly where available; never claim nonexistent
  NEW200 fp32 archive equality. Preserve all originals.
- Historical AUTO from P2-A rows is reused; PATH5 runtime y_A/mask is the all300 adaptation input.
  FIXED100 teachers agree with original TTA1 ancestors; NEW200 is PATH5 clean_teacher/valid_mask,
  same frozen theta0 generation/suppression. No new AUTO decode or language-ID call.
- `cached.Branch` plus core_p1.processed_argmax, branch_adjudication.processed_log_probs and
  path_decode.stream/edit_distance: reusable scoring/cache/token primitives. **Do not use score_branch
  unmodified for STEP1 geometry:** it checks common-prefix greediness; MECH0 deliberately teacher-forces
  the same historical prefix under STEP1 even if that state's earlier greedy path differs.
- Canonical evaluator and P2SEQ/TTA/PATH paired count-sum dialogue bootstrap are reusable after seal.
  Independent auditor must write its own diagnostic/label arithmetic, no primary decision imports.

## Planned additive files

Later add `experiments/inference_cf_p2a2_mech0.py` (prepare/manifest/run/seal),
`inference_cf_p2a2_mech0_analyze.py` (reference-free primary + guarded secondary),
`inference_cf_p2a2_mech0_audit.py` (independent pre/primary/post),
`tests/test_inference_cf_p2a2_mech0.py`, `slurm/inference_cf_p2a2_mech0.sbatch`.
Output root `results/inference_cf/p2a2_mech0/`, canonical rows000..299; selected phase2 rows retain
FULL300 indices, not compact reindexing. Later report `docs/inference_cf/P2_A2_MECH0_REPORT.md` only
following valid output seal/audit. Do not generate report during design or change historical reports.
Only permissible edits to existing A2 code are opt-in detached snapshot observer plumbing described below.
Forbidden edits: A2 objective/optimizer/precision/masks/reset, model/tokenizer/suppression, historical panels,
core METHOD_CONTRACT, G1/G1A/controller and historical scientific results.

## Exact execution sequence

1. Verify clean pushed freeze, all source/audio/model/tokenizer/teacher/output/archive hashes and
   fixed-ID role metadata (only utterance_id/dialogue_id/role). Derive strata independently using
   content decision streams and EOS metadata,44 changed/12 EOS/32 content; controls first2 SAME per
   dialogue40. No references/outcomes/errors/POI labels. OTHER_INVALID or control shortage means STOP.
2. Prepare a resolved plan with runtime allowlist (audio, y_A/mask, fixed settings) and separate diagnostic
   historical common-prefix/donor metadata. No reference fields. Commit/push plan and implementation/tests.
3. Add opt-in observer after each original opt.step, cloning detached fp32 masters to CPU. Do not
   materialize resident weights or decode until both updates finish. Observer calls no forward/backward,
   RNG, loss or optimizer operation; reductions use detached copies. Default original path unchanged.
   Synthetic off/on exact losses/gradients/final arrays; independent live audit unchanged tolerances.
   Config instrumentation_provenance identifies the two permitted files: their source_sha256 entries are
   pinned ORIGINAL baselines, not hashes required of instrumented runtime files. Preserve original Git
   blobs, independently inspect only the allowed diff, and pin NEW runtime hashes in the resolved manifest.
   All other inherited source anchors remain immutable; do not bypass an anchor failure without this review.
4. Independent runnable audit PASS_TO_P2_A2_MECH0. Manifest via provenance/specfreeze conventions pins
   resolved config, source/environment/Git/model files and originals; push manifest before sbatch.
5. One resident model is sufficient. Phase1 canonical300: frozen encoder once, ordinary theta0 decode,
   original uninterrupted A2 two updates/L0L1L2; snapshot1/2; final ordinary A2; all300 original tokens/
   text/termination/state hash; fp32 final archive equality100. Reset in finally. Save selected84 encoders
   losslessly on CPU plus snapshots, reduce all300 per-tensor/module/layer/family norms on detached copies.
   Save LN-only compressed snapshot archives for independent checks, no full model/encoder/KV/vocab dumps.
   Original non-LN bytes and no-param-grad flags must pass. Stop on any mismatch, before phase2.
6. Phase2, only after300/300 barrier: selected84 states; materialize STEP1 then three DROP states in
   fixed order DROP_SELF/DROP_CROSS/DROP_POST. Fresh ordinary free decode each, reset after each.
   FULL_A2/B0 are phase1 outputs. Drop exactly one family from final A2, never cumulatively drop families,
   use theta0 bf16 replacement and exact complement hash. No optimizer or additional adaptation.
7. Changed44 geometry under STEP0/STEP1/STEP2 plus each DROP: same historical prompt/common prefix
   forced token-by-token, fresh state-owned cache. Obtain current query logits, valid candidate/EOS
   logp/entropy/top-gap, action classification; STEP1 prefix mismatch is expected diagnostic metadata.
   For original three states only, teacher-force final A2 H3 content on that same path and score,
   H_eff0=None. Geometry futures are diagnostic donors, not adaptation or free-decoder input features.
8. Seal/push all300 state/loss/norm/reconstruction rows, selected outputs, geometry/continuation/AUTO
   alignment, snapshot archives, cache/reset evidence and manifest/source hashes. Independent primary
   audit PASS before references; a partial panel/seal cannot open evaluator. No thresholds/method selection.
9. Post-seal evaluate canonical FULL300 A2−B0 and subset ablations/STEP1 descriptively. Reproduce historical
   aggregate baselines, compute positive and net termination contributions, dialogues/concentration/LODO,
   paired bootstrap, family action retention and frozen mechanism labels/readiness. Independent post-audit
   P2_A2_MECH0_AUDIT: PASS, commit/push report/current docs. STOP regardless of label. No confirmation run.

## Snapshot and state storage

theta0 snapshot once; fp32 STEP1 snapshots all300; fp32 STEP2 NEW200200; FIXED100 original final archives
reused by hash. Ordered keys use exact194 trainable enumeration; each snapshot under1MiB uncompressed.
STEP1/2 effective hash uses actual bf16 cast and original tensor_bytes_hash. All archives and compact
rows/hashes must be durable on GitHub before reference evaluation; do not leave diagnostic state only local.
Caches are destroyed before any LN materialization/reset. Detached encoder may share across states,
but decoder KV never does. Selected84 encoders need not be serialized/committed once phase2 finishes.

## Analysis definitions and tests

Config/spec are authoritative for all formulas. Test positive-rescue denominators vs net changes explicitly;
undefined fractions are None. Disjoint A2 strata sums must reproduce FULL300; AUTO cross strata form a complete
partition too. Controls do not get a fictitious first-divergence site. Allstate FULL action retention=1 is a
sanity check, and25% family sensitivity is inclusive, never a parameter-selection rule.

Test original two-step trajectory with observer; step1/step2 snapshots distinct and post-step timing;
194 names and64/64/66 family partition; L2 squared aggregation; ablation replacement/complement bytes;
state/cache/reset flags; canonical EOS/cap/indexing; STEP1 forced-prefix mismatch accepted;
finite optional candidate handling, H3/H0 and AUTO positional-history distinction; output seal firewall;
error decomposition/undefined fractions; mechanism precedence and equality boundaries; confirmation-ready
NO on missed FULL_A2 severe truncation, not on diagnostic-state failure. Existing A2/TTA1 CPU suites remain required.

## Interpretation limits

EOS_RECOVERY is defined by a baseline EOS and final A2 content action. A positive step2 stop-margin shift
is therefore partly implied by that group definition, not independent evidence of the mechanism. Retain the
requested criterion as a consistency check; interpret step1 magnitude, continuation support, causal family
ablations and error contribution/concentration together. State-family ablations address the selected parameter
state, not proof that EOS accounts causally for every rescued reference error. No STEP1/trainable/objective selection.
Readiness YES recommends a separately frozen exact-A2 confirmation, never authorizes use of D-dev-confirm/test here.
