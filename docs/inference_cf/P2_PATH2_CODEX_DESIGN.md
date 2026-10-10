# P2-PATH2 — implementation handoff

Design-only freeze. Read P2_PATH2_SPEC.md/config/panel before implementing. No scientific PATH2 code or outcomes were
created here. All choices, thresholds, EOS boundary behavior and terminal precedence are frozen; no new science delegated.
Core v6 is separate and unchanged; preserve original A2/A3/A4/PATH0/PATH1 source/config/report/result bytes.

## Reuse and additive files

- `script_safe_tta.adapt_a4`, `episodic_tta.LNGuard`, `soft_auto_tta.auto_prompt`, A4 sealed plans/checkpoints: exact
  original reconstruction194 tensors/248320 scalars/two updates/teacher/no-op/reset. Reuse PATH1's phase1 barrier,
  native-language identity, first-row independent loss/gradient check and effective-checkpoint equality. No new loss.
- `experiments/inference_cf_cached.py::Branch`: model-owned incremental prompt then one-token feeding, absolute cache
  positions, passive16 capture/attention. `core_p1.processed_argmax`: exact absolute-index suppression. No generic
  HF generate, batched prefix shortcut or different decoder path for causal systems.
- `branch_adjudication.score_branch`, `processed_log_probs`, `consensus`: unchanged H=3 full-vocab mean log-probs,
  0.5 weights, tie1e-12→B. Its returned prefix/position/fed/state flags must all be checked; false traces cannot be ignored.
- `path_decode.stream/edit_distance`, `episodic_tta.severe_truncation`: unchanged ED/EOS/severe marker conventions.
- PATH1/PATH0/A4 immutable seals/audits/raw rows plus PATH2_PANEL: prepare independent identities and C score targets.
  Historical PATH1 U AUTO donor scores have no operational use here. No new U induction/clamp experiment.
- Canonical evaluation modules and `experiments/inference_cf_p2seq_analyze.py` count/outside primitives; reuse existing
  A4/PATH0 comparator provenance/text/tokens and A2 outputs. New audit must independently derive counts/labels from
  canonical metric primitives, not import primary decision logic. No bootstrap or significance gate is needed.

Add `src/csasr/inference_cf/consensus_guard.py`, `experiments/inference_cf_p2path2.py` (prepare/manifest/run/seal),
`inference_cf_p2path2_analyze.py` (reference-free sanity; separately guarded reference outcome),
`inference_cf_p2path2_audit.py` (pre/primary/full), `slurm/inference_cf_p2path2.sbatch`,
`tests/test_inference_cf_p2path2.py`. Prefer reuse without modifying any historical scientific modules. New output root
`results/inference_cf/p2path2/`, run1 with per-row compact JSON; later `docs/inference_cf/P2_PATH2_REPORT.md`.
No outcome report now. Pin runtime/additive code and dependency/model hashes in the resolved manifest before submission.

## Safe online engine

Load two independent local bf16/eager bundles, theta0 and adaptation bundle. Verify decoder-LN storage does not alias,
non-LN model identities/config/tokenizer/precision/suppression match original artifacts. Theta0 is permanently frozen;
A4 instance is reconstructed once per row and held unchanged while any decoder path is alive. No weight-switch scheduler
with persistent caches. Detached encoder output from one frozen encoder can be shared; decoder KV can never be shared.
Test encoder sharing bit identity and ensure clone for adaptation is outside inference_mode as in PATH1.

Phase1 builds12 effective states/detached encoders through original reconstruction, verifying all12 theta0 B0, FREE-A4,
checkpoint, teacher/language/reset before proceeding. No controller row starts before all-row barrier. Reset/materialize
only after all live decoder caches have been destroyed. Record instance/owner/state hashes and token/position traces.

Online detector owns fresh theta0 and A4 Branches and advances them from identical content histories. Compare next-token
processed argmax at absolute t before emission. Feed equal non-EOS to both; equal EOS ends, cap limits output decisions.
At first mismatch return live prefix/argmax with no historical site/group access. Discard detector caches before short
candidate/score/controller paths to keep ownership unambiguous. Audit expectations applied outside runtime API.

Generate each candidate from a fresh owner cache, replaying prompt once and live prefix token-by-token, then greedy
until3 content tokens, earlier EOS or global budget. EOS-first mismatch INVALID for this panel; no altered score.
Score four fresh paths via unchanged PATH1 scorer; consume suppressed-token checks, owner locks and prefix checks.
Pass the SAME immutable decision object (prefix, candidates, scores, winner) to G1/G2; neither may choose again.

G1: fresh A4 path, greedy pretrigger decisions must equal live prefix, selected first token then ordinary A4 until
EOS/cap. Cache created/advanced entirely under A4. This is equivalent to an A4 one-token clamp with online token/site,
not a PATH0-donor lookup. G2: A4 winner → ordinary A4 unchanged; theta0 winner → commit actual theta0 greedy candidate
history, destroy candidate caches and reconstruct a fresh A4 cache from full prompt+selected history incrementally.
Resume from logits following the last selected token. Do not duplicate tokens during replay, reuse detector/rollout
KV across owners or pretend replay changes model state. If handoff terminates/exhausts budget, output stops without
an A4 release. Postguard never compare argmax for another trigger.

All C expected trigger sites/greedy branches correspond exactly to original B0/A4 donors. Require byte-equivalent
serialized logprob values and score means/margin/winner against PATH1; only labels/owner IDs may differ. This follows
from identical common history, greedy states and scoring path; a discrepancy is invalid, not evidence about the guard.
No-trigger U7 and A4-winning C2 must each give G1=G2=G0 exact IDs/termination. Do not special-case them from group/winner
expectation: execute/audit general controller semantics. Cached baseline outputs are comparison anchors, not runtime
features. Old RESCUE continuations are not substitutes for new online controller execution.

## Focused test contract

Reuse PATH0/PATH1/A4 tests. Add only controller tests needed here:

1. Exact12 IDs/order/raw D inequality, C5/U7 analysis split and parent/source/panel fingerprints. No scientific
   resiting: recompute historical first divergences independently; all5 C H_eff3, U no-trigger expectation audit-only.
2. Exact original A4 teacher194 tensors/2 updates/reset/precision/checkpoint; theta0 immutable independent instance,
   non-LN unchanged/no LN storage alias, no parameter grads retained. All-row reconstruction barrier rejects one mismatch.
3. Online first **any-token** processed-argmax mismatch only, exact shared emitted prefix, begin suppression at
   absolute0 only, tie argmax from existing implementation; one guard max; common EOS/cap no-trigger.
4. Fresh model-owned cache per path; forbidden cross-model cache fixture rejected; state immutable while live;
   detector/candidate/score/G1/replay cache ownership and disposal. Contiguous positions/fed history asserted. Mutating
   one instance must not change the other's LN. Equal cache lengths alone insufficient as a test.
5. Live candidates greedy under theta0/A4 (not AUTO/donor), H3 shortened correctly at EOS/cap, no EOS scoring or
   padding, EOS-first invalid. Absolute total output cap200; candidate/replay work must not consume emission budget.
6. Exact PATH1 four scores, full-vocab processed float32 log_softmax, mean H_eff, .5/.5 consensus/tieB1e-12. Per-token
   C scores/candidates/k/margin/winner equal historical PATH1. U historical AUTO candidates must never be scored.
7. G1 commits exactly winner first token and advances A4; G2 commits theta0 H3 only on theta0 win, then FRESH A4
   prompt/history replay; A4 win/no-trigger exact no-op. Assert identical shared decision hash/winner for G1/G2.
   Include toy fixture where forcing1 and theta03 histories lead to different A4 suffixes, and replay/fresh A4 clamp
   with identical full history yield the same logits (history-only interpretation).
8. Reference-free controller input allowlist rejects references, C/U, frozen sites/donors, AUTO outputs, evaluator
   columns. Construction invariant when those audit/analysis fields are changed; only audit assertions may fail.
9. ED logical EOS/termination, severe marker boundaries, all12 count cutoffs inclusive29/105/143, zero caps/severe,
   2-error material advantage OR condition/POI slack1. Independently enumerate both pass booleans and material flag:
   valid label precedence is exhaustive, including G1 pass/G2 unsafe raw advantage. Invalid overrides all outcomes.
10. Seal/hash/committed-remote barrier prevents new reference evaluation before pushed immutable outputs and primary
    audit. Independent auditor import scan rejects primary decision code. Historical comparators hash mismatch invalid.

Contract-only checking now verifies source/panel hashes,12 C/U source identities, historical count cross-check,
cutoff derivations, all five C branch donors/scored inputs and complete terminal truth table. No full-model tests,
PATH2 adaptation or controller outcomes in this Codex session.

## Independent audit and run order

1. Implement additive code/tests only; run focused CPU tests, review diff, commit/push. CPU prepare recomputes exact12
   identities, parent raw token/site/hash checks, C exact-score targets, U no-trigger expectations and comparator map;
   preserves sealed source hashes. No reference loading or reference-derived runtime features.
2. Independent pre-run audit verifies implementation/cache ownership, immutable A4/score semantics, all budgets,
   numeric count derivations and branch/label truth table. Require **PASS_TO_P2_PATH2** committed/pushed. Freeze resolved
   manifest git/environment/model/tokenizer/code/input hashes, trainables/suppression and one-job30min budget; push.
3. One sbatch allocation: barrier reconstruction12/12 first; then online detector, live short candidates/scores,
   G0/G1/G2, reference-free traces and exact C/U/no-trigger/winner identities.24 adaptation updates<=25 backwards;
   no-trigger rows have no branch scoring; no extra objective or scientific reruns. Partial/budget/identity failure
   INVALID STOP, no fallback/exclusion or automatic second job.
4. Seal all transcripts/termination/triggers/candidates/scores/winners/traces/ED/manifest hashes, push immutable seal.
   Independent primary audit rechecks online event, own-model caches, no stale replay, exact PATH1 reproduction and
   all no-op identities. Require primary-phase PASS before new PATH2 reference evaluation.
5. Canonical B0/AUTO/A2/A4/G1/G2 evaluation after barrier, exact29/105/143 outcome bounds and frozen label, all12 and
   C/U/per-row safety/benefit interpretation. Independent full **P2_PATH2_AUDIT: PASS**: recompute population, argmax,
   score arithmetic/winner, reconstruction, cache ownership, count cutoffs/material advantage/final precedence and
   reference barrier without importing primary decision implementation. Commit/push report/state and STOP.

A positive label authorizes no follow-on automatically. No full24/100/300/P3, new objective, repeated guard, horizon/
LR/parameter sweep or fresh validation. One GPU job12 rows,target<10min,hard30min; no parallel PATH2 jobs. Final tree
clean and local HEAD equals origin/cs-asr-steer-inf. All important design/run state must be pushed.
