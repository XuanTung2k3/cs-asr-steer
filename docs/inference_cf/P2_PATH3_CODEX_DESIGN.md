# P2-PATH3 — implementation handoff

Design-only freeze. Read SPEC/config/panel; implement exact A4 + PATH2 G1 on fixed100, without altering historical
modules or outcomes. No PATH3 runner/outcome was created or run here. Core v6 is separate/unchanged.

## Reuse map and additive implementation

- `consensus_guard.lockstep_detect/rollout/g1_forced`, `branch_adjudication.score_branch/consensus/owned_clamp`: exact
  PATH2 first-disagreement, H3 mean generation-processed score/tie, one-token A4 execution. Pin byte hashes; no edits.
  Do not invoke g2_forced or any handoff. Existing helpers return false trace checks: callers must reject all failures.
- `script_safe_tta.adapt_a4`, `token_classes`, `episodic_tta.LNGuard`, `soft_auto_tta` teachers, original A4 optimizer:
  unchanged teacher194 affine tensors/fp32/bf16/two updates/reset/no-op. All100 newly computed, no cached A4 substitute.
- `experiments/inference_cf_p2path2.py`: reuse two-instance/no-alias/non-LN checks, encoder clone, reconstruction,
  first-row independent live check, state hashing, manifest and online decision construction. Extend population in
  an ADDITIVE new runner, not by changing PATH2 contract/code. Parent-specific C/U/PATH1 assertions are restricted
  to the12-row replay comparator; they are not runtime policy or requirements on other88.
- `cached.Branch`, `core_p1.processed_argmax`, `path_decode` content/EOS/ED, `episodic_tta.severe_truncation`: exact
  same incremental execution path, absolute-index suppression, cap and termination. No HF-generate alternative.
- `P2_SEL_MINI_PANEL.json`, TTA1 plan/seal + parent A3/A4/PATH2 panel/plans: exact identities, teachers/masks, audio
  fingerprints/comparators and old checkpoints. New panel is a fingerprint wrapper, not a new selected panel.
- `experiments/inference_cf_p2tta_a3.py::resolve_a2`: reused A2 = TTA1 new80 rows + TTA0-R audited20 ancestor rows;
  panel pins correct source indices/selectors/hashes. No A2 rerun. B0=P2SEQ S0, AUTO=historical B0_AUTO. Existing output
  text/IDs/termination and provenance allowed; do not load historical per-row ASR counts as preparation inputs.
- Canonical/pier/mer/retention plus P2SEQ count/outside-harm/draws/ratio primitives; same TTA1 safety and denominator
  conventions. Independent auditor shares metric primitives only, never imports new primary decision code.

Add `experiments/inference_cf_p2path3.py` (prepare/manifest/run/seal), `_analyze.py` (reference-free summary plus guarded
post-seal evaluation/decision), `_audit.py` (independent pre/primary/full), `slurm/inference_cf_p2path3.sbatch`,
`tests/test_inference_cf_p2path3.py`. No new controller module is needed. Outputs `results/inference_cf/p2path3/run1`,
canonical_index per-row files000..099, immutable plan/preaudit/manifest/seal/primary/secondary audits. Later
`docs/inference_cf/P2_PATH3_REPORT.md`; no outcome report now. Minimal current docs/plan updates after implementation.

## Reference-free preparation

Verify all source/model/tokenizer/environment hashes and independent IDs/dialogues/order before model execution.
Parent file SHA and internal panel hash differ in domain; compare the correct byte fingerprint, do not regenerate
selection. Verify100/24/12/76 strict nesting. Keep original100 canonical order; job order is PATH2 native12 then
remaining88 canonical. Membership/partition fields only logging/analysis/replay routing, never guard inputs.

Runtime teacher rows from original TTA1 plan: y_A, valid_mask, audio path/hashes. Compute original canonical token_classes
reference-free; verify exact original classes/mask/language for DEV24. Detect native AUTO language from theta0
input_features path on each row as original A4; do not infer it from historical text, use a different LID provider or
regenerate AUTO pseudo transcripts. Old checkpoints exist only for DEV24; never invent expected new76 parameters.

Keep controller inputs strictly encoder tensor, two model states, prompt/generation config, live token history and
owned caches. Separate runtime/A4 reconstruction inputs from audit targets (historical outputs/sites/partition).
Freeze comparison-map metadata and relevant source hashes in new plan; no reference/error/evaluator module import.

## Job sequence and hard replay barrier

One job loads two independent bf16/eager models once. Non-LN weights are identical and immutable, LN storage distinct;
theta0 never changes. Original encoder output once per row/detached clone, fresh masters+optimizer per episode.
No weights mutate while any path/cache exists; destroy all caches before reset/materialization/next utterance.

First12 ONLY: reproduce PATH2 phase1 unchanged A4 reconstruction (24 updates and original first-row live check),
B0/theta0 sanity, checkpoint bit identity and original G0. Then original G1 online pipeline for12, omitting G2 calls.
Compare full shared scientific fields against PATH2: detector/prefix/argmax, b0/b4/H_eff/termination, per-token scores/
means/consensus/margin/winner/decision_hash, G0/G1 IDs/termination, state/reset/owner sanity. Exact comparison ignoring
new schema owner labels/manifest/timing counters only. No scientific tolerance loosened. All12 PASS before any encoder/
adaptation/controller computation on remaining88. Partial mismatch abort INVALID, no correction to parent modules.

After barrier, remaining88 sequentially under same reconstruction + G1, fresh episode/reset. For DEV24_OTHER compare
A4 against original state/checkpoint/native condition/free output. For NOVEL76 no historical A4 expectation; original
semantic/finite/reset/teacher/state-owner checks suffice. No assumption that these rows never trigger/no-op.
No extra full theta0 decode outside replay12 merely for convenience: use sealed B0 comparators and online dual calls.
No-trigger output is verified/copied from that row's live A4 (not historical B0); A4-winner G1 must equal live A4.
EOS-first trigger remains inherited INVALID on the broader panel; do not add a new branch score or silent exception.

For triggered rows live H3 rollouts/four score paths/G1 identical to PATH2. Each decoder path fresh cache under fixed
owner; first emitted winner token consumed under A4, release immediately, never compare/guard again. Keep resulting
G1 diagnostic decision hash, traces and owner hashes; no G2/handoff or policy fallback. Store both live A4 and G1
content IDs/text/termination and compact diagnostics, no unnecessary full hidden/teacher-distribution dumps. Loss/
state/reset scalar hashes and original live-check evidence required; historical DEV24 checkpoint comparison explicit.
Counters must reflect exactly200 optimizer updates,<=201 backwards, zero new controller backward calls, one guard max
per row, two rollouts/four score paths per trigger, no scores on no-trigger rows.

## Focused tests and audits

Reuse existing PATH2/A4/PATH1 tests. Add contract/coupling tests only:

1. Exact fixed100 ID/dialogue/order/hash, source audio/B0/AUTO/A2 provenance, strict12⊂24⊂100, NOVEL76 correct difference,
   partition membership hashes and execution-vs-canonical order. Reject refill, duplicate/missing row or renamed role.
2. Parent A4 trainable names/optimizer/two updates/fp32/bf16/mask/classes/no-op exact; parent G1 helpers unchanged
   code hashes. NOVEL76 adaptation starts theta0, no episode/master/optimizer/cache transfer. No reference features.
3. Replay12 scientific value projection exact (including float logprobs/margin), reset/hash/token/termination; one
   injected mismatch blocks remaining88 before encoder forward. NO_TRIGGER null decision; no old C/U logic on new rows.
4. Same trigger/tie/suppression/H3/caches/one selected token/release/one-event max; stale cross-model KV fixture rejected.
   No g2/handoff calls in new runner. EOS/cap behavior inherited, empty-EOS branch invalid, no score fallback.
5. Safety exact TTA1 subset; rate denominators/None semantics and severe truncation/cap delta boundaries. Float1e-12
   inclusive safeguards. All100 output/safety fields complete before label; empty primary language population invalid.
6. Rescue formula integer boundaries including X=0/negative, odd positive excess and ceil; require>=1 even absent
   A4 excess. Benefit signed counts with I_A4<=0 guard, I_G1 strictly>0, retention>=.75, MER_G1−A4<=.005.
7. Breadth positive/net decomposition, positive-rescue dialogues/concentration, R_plus=0→C=1, inclusive.50/.60,
   net1/min3 rows/min3 dialogues. A toy single-row rescue of20 never passes; DEV24/PATH2 rescue cannot enter NOVEL76.
   Include positive rescue with offsetting novel harm, concentration boundary, same-dialogue three-row failure.
8. LODO includes all20 represented NOVEL76 dialogues (including zero-rescue), strict>0 and90%→18 removals; secondary
   flag never changes label. Bootstrap deterministic paired blocks2000/seed240924, count-sum rates and shared draws.
9. Exhaustive label precedence INVALID→DAMAGE→NO_GAIN→OVERCONSERVATIVE→CONCENTRATED→SUPPORTED; early failures override
   later passes/LODO/CI. Neither statistical significance nor full100 original rescue alone can yield SUPPORTED.
10. Reference/evaluator barrier requires pushed immutable100 outputs and primary audit; reject missing/changed file,
    historical evaluation-count access in runtime/prepare, and unsealed reference load. Auditor import scan independent
    from primary decision code. Comparative counts/retention use canonical primitives after seal only.

Pre-run independent **PASS_TO_P2_PATH3** checks source hashes/panel/partitions, teacher/A4/G1 inheritance, cache ownership,
replay barrier, safety/formulas/breadth/concentration/LODO/label/compute/firewall. Post-seal primary audit verifies all100
reference-free diagnostics and exact12 replay/no-op/owner checks. Full **P2_PATH3_AUDIT: PASS** independently recomputes
canonical counts/safety/gain/retention/breadth/LODO/paired-bootstrap/labels; no import of primary decision logic.
Do not launch a second GPU job for an audit or failure fallback. Live/synthetic independent checks plus sealed raw
outputs and source/trace audit reuse the established approach; audit cannot be replaced by primary report assertions.

## Exact execution order for Claude

1. Implement additive runner/analyzer/auditor/tests, no parent science edits. CPU prepare verifies pinned source hashes,
   identity and comparator/teacher map; run focused tests, inspect diff, commit/push.
2. Independent preaudit PASS_TO_P2_PATH3, push audit + resolved manifest/config/code/model/environment/source hashes.
3. One sbatch allocation (100 episodes, target<15min,hard30min): replay12 exact barrier then remaining88. No G2/A3/new
   baseline outcomes; no repeated guard, new policy or expanded data. Any runtime/budget failure INVALID STOP.
4. Reference-free100 output/controller-summary seal, commit/push; independent primary audit PASS before references.
5. Canonical FULL100 all5 and partition A4/G1 evaluation; point gates and NOVEL76 breadth, LODO/CI descriptive only;
   independent full P2_PATH3_AUDIT: PASS; commit/push report/status/exposure/plan and STOP regardless of verdict.

SUPPORTED only recommends separate independent confirmation. No automatic full300/P3/transfer/fresh-validation/
new-panel stage. Final clean tree/local HEAD=remote; preserve every historical outcome and failed attempt.
