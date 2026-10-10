# P2-PATH2 — Online Consensus Guard + Short theta0 Handoff

Pre-outcome controller diagnostic, following audited `P2_PATH1_MIXED`. Starting local and fetched remote HEAD
`0b2e638498387b9dcef55941f2fc1ccfd15906e1`, branch `feature/inference-cf-steering`, clean tree, empty user queue.
This session freezes documents/config only: no PATH2 inference, adaptation, controller outputs or scientific job.
Normative artifacts: this spec, `P2_PATH2_CODEX_DESIGN.md`, `P2_PATH2_PANEL.json` and
`configs/inference_cf/p2_path2.json`. Core v6/METHOD_CONTRACT is unchanged; this is separate exposed-development
supporting diagnosis. Historical A2/A3/A4/PATH0/PATH1 remain immutable and closed.

## Question and claim boundary

Can the fixed PATH1 consensus adjudicator prevent harmful entry while preserving useful A4 switches? Is one selected
token sufficient (G1), or is a three-content-token theta0 handoff necessary (G2)? PATH1 found both prefix causality and
adapted-state amplification. This is one online controller event, not a new objective, repeated guard, beam search,
threshold/horizon sweep, or deployment validation. G2's lasting effect is the selected three-token **history**. After
replay its decoder cache is entirely A4; there is no latent theta0 memory or independent hidden-state rollback.
The legacy terminal label `ROLLBACK_NEEDED` below means short handoff/history, not a hidden-state rollback claim.

## Population and provenance

Exactly the twelve PATH0 D utterances (theta0 AUTO differs from theta0 B0-FORCED), same order and source hashes. The
new panel fingerprints original PATH0/PATH1 files and each row; it retains sealed raw B0/AUTO/A4 arrays, checkpoints
and five C audit expectations. Independently comparing raw content arrays **and termination** gives C=5 (A4 differs
from B0), U=7 (A4 equals B0), nine dialogues. C/U and expected sites are analysis/integrity metadata only, never
controller inputs. All twelve rows retained; no new selection, refilling or difficult-row exclusion.

Panel byte SHA256: `ade7567c1e13f75e410e6a1cf27a3a613bbe9a9ac5d5898fa9dc18f8ea3d0923`.
Panel derives no labels/sites from references. Historical exposed per-row canonical counts may be inspected now
solely to freeze the outcome cutoffs; no new PATH2 reference evaluation occurs before the later output seal.

## Unchanged A4 and decoding

Reconstruct original A4 exactly once per row through unchanged `script_safe_tta.adapt_a4`: original sealed theta0 AUTO
teacher/forced teacher, y_A/mask/classes/language, script-safe forward KL; 194 decoder-LN affine tensors/248320 scalars,
fresh fp32 masters and AdamW lr=1e-3, wd=0, betas=(0.9,0.999), eps=1e-8, foreach/fused=False, exactly two updates,
no scheduler/clipping/scaler, original no-op rule, exact episodic reset. Old AUTO information is used only by unchanged
A4 reconstruction, never as a runtime candidate or consensus teacher. No steering, D2, LID or additional objective.

All decoding/scoring uses the same frozen local model/tokenizer, bf16/eager/eval/batch1, forced-ZH prompt
`[50258,50260,50360,50364]`, cached.Branch prompt once then single-token steps, passive L16 capture and attention=True,
processed_argmax with the copied suppression arrays; begin suppression applies only at absolute content decision0.
No sampling/beam; cap200 output decisions including a terminating EOS decision. EOS=50257 is not stored as content;
record `terminated=eos|cap`. Token ED appends a single logical EOS iff eos-terminated; no text retokenization. All
observed parent content IDs must remain unchanged, including UTF8 fragments. Unexpected generated prompt/control or
timestamp IDs versus the historical content allowlist, nonfinite logits, or suppression violations are INVALID.

Before any controller intervention, an all-row barrier requires: theta0 ordinary decode=B0 tokens/termination12/12;
reconstructed FREE-A4=sealed A4 tokens/termination12/12; effective bf16 LN tensors equal sealed final fp32 masters cast
bf16; language/teacher identity and exact reset/non-LN invariance. Reuse first-row independent loss/gradient check
(1e-5 loss,2% relative gradient, floor1e-8), not an extra update. Stop on mismatch; do not repair/retrain A4. G0 is this
ordinary reconstructed FREE-A4 baseline. No cached historical prediction can substitute for the live reconstruction
sanity. Small effective states and detached encoders may be retained after the barrier.

## Cache ownership and online trigger

Use **two independent resident model instances** with identical frozen pretrained non-LN weights: immutable theta0
and reconstructed A4. No decoder-LN storage alias between instances. A4 LN state stays fixed throughout all paths for
that row; theta0 never changes. Share only detached frozen encoder outputs (encoder parameters identical, compute once
per row), not decoder KV. Each Branch owns an instance/state hash; start cache=None and feed contiguous positions.
Independent candidate and scoring paths have distinct fresh caches. Dispose all row caches before A4 reset/next row.
Record instance/owner IDs, LN hash before/after, consumed tokens, positions and cache length; source inspection is also
required because a trace/hash alone does not prove ownership. No weight switch while a live path/cache exists.

Online before a guard, feed the same fixed prompt and emitted content token history to both instances, separately.
At each absolute content decision compare their canonical processed argmax IDs. Equal non-EOS: append/feed that token
to both under their own weights. Equal EOS: ordinary termination, NO_TRIGGER. First unequal IDs: trigger at k and
freeze the live common prefix; assert exact prefix identity and both caches generated entirely under their own state.
Only one guard per utterance; never run another disagreement decision after it. This is the first **any-token**
disagreement, without a language classifier. No historical k or group may schedule/suppress it.

If no disagreement before termination/cap, G1=G2=G0=A4 token-for-token and termination-for-termination. Expected U7 all
NO_TRIGGER, mandatory. Expected C5 trigger at historical first A4/B0 divergence, mandatory: k=0,5,15,0,10 in parent
C order. Historical expectations are compared after the online computation, never fed into the controller.

## H=3 live branches and exact PATH1 score

At trigger, independently replay prompt + live common prefix under each owner using token-by-token Branch, and roll
out up to H=3 greedy **content tokens**: b0 under theta0, b4 under A4. No historical donor/AUTO-conditioned generation.
The first branch IDs must differ and equal their respective trigger argmax. Both branches must have at least one
content token. Shorten independently if EOS or global cap is reached before three content tokens; retain termination
metadata, never fabricate/score EOS or pad a branch. No lookahead beyond the requested three content tokens. Horizon
also respects remaining global output budget, counting all committed output decisions, never replay work.

If a trigger's first candidate is EOS, the exact PATH1 content-only mean score would have an empty branch. Such an
unexpected case is INVALID on this fixed panel (all five historical triggers have three valid content tokens); no
fallback EOS score, silent NO_TRIGGER or new scoring convention. This diagnostic does not claim unrestricted EOS
branch handling. EOS later inside rollout shortens the scored branch. If a selected G2 branch then terminates, stop
there rather than forcing three non-EOS tokens or inventing a release. Cap exhaustion likewise stops at the global cap.

Score both branches under both models using unchanged `branch_adjudication.score_branch`, fresh owner-specific cache
per score, prompt/common prefix replay and per-token scoring before consumption at absolute k+j:

`S_m(b) = sum_j log_softmax(processed_float32_logits_m(k+j))[b_j] / H_eff(b)`;
`S_cons(b) = 0.5*S_theta0(b) + 0.5*S_A4(b)`.

Temperature1, full allowed vocabulary, same suppression; no gradient. Larger score wins. Absolute difference<=1e-12
chooses b0/theta0 (PATH1 B tie rule). Store four score paths and per-token log probabilities. H_eff is per branch,
EOS excluded: mean normalization does not assess termination likelihood, a preserved limitation. No confidence
threshold, alternative S_min adjudicator or score search.

**Exact C reproduction:** online common prefix and k are identical to PATH0/PATH1. Under the same greedy states the
live b0/b4 are exactly sealed B0/A4 `[k:k+3]`, hence exactly PATH1's C donors. All five branches have H_eff=3. Require
exact token IDs/bounds, serialized per-token float32 log probabilities, each four mean scores, consensus margin and
winner against PATH1 run1. Ownership labels/counters may differ; numerical scoring semantics may not. Expected winners
are A4 for U0021_S0_513/U0023_S0_666; theta0 for U0023_S0_87/U0023_S0_664/U0092_S0_101. These are audit expectations,
not a runtime lookup. No relaxed post-outcome tolerance. U historical AUTO-donor scores are **not** reproduced:
U has no online disagreement, so it must not receive branch scoring or an AUTO clamp.

## G1 and G2

Compute one online trigger/candidate/score/winner artifact per row and share that exact decision between G1/G2. Neither
may recompute a winner from its changed continuation. Prefix before intervention must match G0 and both live models.

- **G1 CONSENSUS-1:** emit winner's first content token only, consume it under A4 and immediately release to ordinary
  A4 greedy continuation. Cache is A4 from prompt through selected token. No theta0 decoder KV is imported. If b4
  wins, G1 must equal G0 exactly. Remaining candidate tokens are diagnostic/scoring only.
- **G2 CONSENSUS-GUARD3:** b4 wins → simply ordinary A4, so G2=G1=G0 exactly; no unnecessary three-token replay or
  handoff. b0 wins → theta0 generates/controls its already-scored b0, append those up-to3 tokens to the common output
  history; discard all candidate caches. For a nonterminated branch with remaining budget, build a **fresh A4** cache,
  feeding fixed prompt once and the entire selected content history one token per call. Assert fed=prompt+complete
  selected history, positions contiguous, owner A4 for every call. Resume from logits after the last selected token.
  Do not emit replayed tokens again, count replay as new output, or batch-prefill the content history. Stop without
  fictitious A4 release if theta0 reaches EOS/cap during handoff.

A4 may already have computed logits at the first decision; those are not reusable after a theta0 handoff. Encoder
sharing is safe; decoder cache sharing is prohibited. Replaying the history deterministically under A4 leaves only a
history intervention. G2 benefit beyond G1 supports a short guarded region overcoming persistent bias, not cache memory.

## Reference-free record and seal

Per row: trigger yes/no/k, live prefix/hash, dual argmax, candidate tokens/H_eff/termination, all four score paths,
consensus scores/margin/winner, G0/G1/G2 content IDs and termination, ED to sealed B0/A4/AUTO decision streams, changed
vs A4, length/EOS/cap, severe-truncation flags, state/cache traces, replay history hashes and counters. No-trigger rows
record no score/winner, never synthetic consensus values. Add manifest resolved config/source/input/model/tokenizer/
environment/git hashes, encoder/state identity, budget/runtime/peak allocated/reserved VRAM, teacher/student/backward/
rollout/scoring/replay/guard counters. Use existing provenance/specfreeze helpers and inference_cf manifest discipline.

Seal all12 outputs, complete primary sanity/diagnostic inputs, panel/config hashes and manifest before new PATH2
references/evaluator access. Commit/push immutable seal first; independent primary-phase audit must pass before
secondary evaluation. Incomplete/failed barriers, unexpected triggers, numerical C-score mismatch, stale caches,
multiple events, no-trigger changes or winner disagreement → INVALID/STOP. No scientific conclusion without full
`P2_PATH2_AUDIT: PASS`.

## Frozen historical counts and controller outcome criterion

Independently summed all12 sealed PATH0 per-row count vectors, cross-checked against A4's original per_utterance
vectors. All denominators agree: 157 POI units,120 ZH units,278 mixed units. These are prior exposed counts only.

| System | ZH errors | POI errors | Mixed errors |
|---|---:|---:|---:|
| B0-FORCED |18|119|137|
| A4 / G0 |41|101|143|

For EACH G in {G1,G2}, full pass is validity AND all of:

1. ZH errors<=**29**: rescue at least half of23 excess errors, requiring ceil(23/2)=12 fewer than A4.
2. POI errors<=**105**: retain at least75% of18-error benefit, requiring ceil(0.75*18)=14 fewer than B0.
3. Mixed errors<=**143**, no worse than A4.
4. Caps=**0**; no new severe truncation. Use existing `episodic_tta.severe_truncation`: baseline content length>=10,
   new output ends EOS, new length<=floor(0.5*baseline length). Check against **both** sealed B0 and A4 per row;
   require zero such G flags. Old A4-vs-B0 severe flags and B0/A4 cap counts are all0, independently checked now.

Evaluate all twelve rows including no-trigger controls; no success requirement based on removing them. Canonical
TTA retention/outside-harm/POI-corruption metrics are reported without imposing an additional unstated gate. This is
an enriched development count screen, not population inference or a significance claim. No bootstrap/CI success gate.

**Raw material G2 advantage** iff (ZH_G1−ZH_G2>=2 OR mixed_G1−mixed_G2>=2) AND POI_G2<=POI_G1+1 AND no new
cap/severe-truncation relative to G1 and B0/A4. POI slack is expressed as net error count, not ambiguous correction
credit. Also report correction/corruption counts separately. A **supported extra advantage** requires G2's full pass
AND this raw material predicate; a safety/benefit-violating G2 cannot displace an already supported G1.

Exhaustive deterministic precedence:

1. Any engineering/integrity/completeness/audit failure → **P2_PATH2_INVALID**.
2. G1 pass AND G2 pass AND raw material G2 advantage → **P2_PATH2_MIXED_GUARD_PROMISING**.
3. G1 pass AND NOT(G2 pass AND raw material G2 advantage) → **P2_PATH2_BRANCH_CONTROL_SUFFICIENT**.
4. G1 fail AND G2 pass → **P2_PATH2_ROLLBACK_NEEDED** (short theta0 handoff, no latent rollback claim).
5. Both fail → **P2_PATH2_LOCAL_GUARD_INSUFFICIENT**.

This also fixes the otherwise uncovered boundary where G1 passes but G2 has a raw two-error advantage while breaching
POI retention: classify supported G1 as sufficient and disclose the unsafe G2 tradeoff. No additional label after outcomes.

## Secondary evaluation and interpretation

After pushed seal and primary audit, canonical evaluation of B0-FORCED/B0-AUTO/A2/A4/G1/G2 on all12 and C/U separately.
Reuse comparator outputs/text only with exact row/source/path/hash/tokenization/metric provenance; no comparator rerun.
Optional A3 descriptive reuse only. Report PIER/MER/EN-WER/ZH-CER, ZH/POI/mixed errors, POI corrections/corruptions,
matrix-ZH/embedded-EN retention, lexical outside harm, D/I/S, caps/severe truncations and per-row counts. Use existing
canonical/pier/mer/retention/outside primitives; forced output text from the frozen tokenizer, no manual normalization.
Retention/harm deltas have B0 causal reference and A4 controller reference explicitly named. AUTO remains practical
comparator only. No significance claim for twelve enriched rows, nine dialogues.

For harmful C rows describe winner and G1/G2 ZH rescue; for beneficial C rows describe retained POI gains; U must be
exact no-op. U0023_S0_664, U0021_S0_513 and U0023_S0_666 are descriptive cases only, never special runtime rules.
A favorable label supports only this one-event controller on this selected cohort. Stop after report/audit; no automatic
full24/100/300, repeated guard, confirmation, new objective or P3.

## Compute, firewall, tests and audit

One sbatch H100/MIG allocation maximum,12 rows, target<10min, hard30min, maxone pending/running PATH2 job. Original24
A4 optimizer updates,<=25 backwards including original independent check; controller/scoring forward-only. Up to one
guard per row, two short candidate rollouts and four short score paths per trigger, none for NO_TRIGGER. Two resident
models permit safe lockstep without state swapping; no extra adaptation. Stop INVALID on budget/partial failure; no
automatic second outcome job or smaller panel. Reuse encoder, old comparators, adaptation and evaluator machinery.

Allowed only already-exposed D-dev-select exact PATH0/PATH1 D12 and old comparator artifacts. Forbidden fresh
validation, confirm/test, router-calib new role, any other development role, full24/100/300, P3 and transfer corpora
(SEAME/CS-FLEURS/ViMedCSS/ASCEND/others). Runtime controller accepts only waveform-derived encoder, frozen model states,
current emitted token history, generation config and model-owned caches. References/C-U/historical sites/donors/
outcomes may not construct trigger/candidates/score/winner/handoff.

Focused tests and independent audit implementation obligations are in CODEX_DESIGN. Before job require committed
`PASS_TO_P2_PATH2`; after seal require independent primary validity; after secondary require `P2_PATH2_AUDIT: PASS`.
Auditor must not import primary decision logic. Every intentional change: checks → reviewed diff → commit → push
`origin HEAD:cs-asr-steer-inf`. Never force-push/merge main/PR; final tree clean and local HEAD=remote. No outcome report
is created in this design freeze.
