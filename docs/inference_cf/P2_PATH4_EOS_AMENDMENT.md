# P2-PATH4-R1 — EOS-boundary extension

**contract_revision = PATH4_R1_EOS_BOUNDARY_V1**. Authorized by the human after reviewing
`P2_PATH4_BLOCKED_EOS_FIRST_BRANCH` at commit `29661e972f0ebfd727d228d2f6dbd9b17dd41c5a`.
The blocker was found before runner implementation, adaptation, scientific output, GPU execution or
opening PATH4 references. The blocked record is preserved verbatim in the original spec/design suffix,
unchanged historical panel, Git commit, and archived `configs/inference_cf/p2_path4_blocked_v0.json`.
Historical BLOCKED text describes revision0; this amendment controls only its EOS-boundary restriction.
No PATH4 scientific outcome or reference evaluation is produced in this amendment session.

## Exact allowed delta

Ordinary CONTENT_G1 is unchanged: first live disagreement, H=3 content rollouts, four model/branch
scoring paths, float32 processed log probabilities, .5/.5 consensus, inherited tie, one selected token
then ordinary adapted A2. Byte-identical `consensus_guard.py`, `branch_adjudication.py`, and
PATH3 `online_g1` remain untouched. Content/content dispatch calls original online_g1 with A2
in its legacy adapted A4 slot. If detection is replayed by that function, verify the same first k,
prefix/proposals; duplicate computation is not a second logical guard event. Require bit-identical
ordinary decisions, scores, tokens and termination on **every content/content case**, both synthetic
regression now and a reference-free direct-original comparator later, from the same saved A2 state.
This comparator is engineering-only inside the same allocation; never updates or changes the method.
No H=3 scientific branch semantics change, EOS later merely shortens an ordinary rollout as before.

At the first live disagreement, exactly one EOS and one generation-valid content proposal invokes
EOS_BOUNDARY, symmetrically for either model orientation. No IDs/sites/labels select this mode.
Let c0/c2 be the two live proposals. At the identical common prefix obtain raw current-query logits
under each owner model. Under the **same** general/begin suppression and absolute index as original
G1, compute float32 processed log_softmax. Score both legal next actions, including EOS:

`s_m(c)=processed_logp_m(c | common_prefix)`;
`S_boundary(c)=.5*s_theta0(c)+.5*s_A2(c)`.

H=1 action for both candidates; no H3 lookahead, candidate continuation, EOS padding or mean over
unequal horizons. Temperature1, full allowed vocabulary, original normalization, no score threshold.
EOS can be scored when legal; it is still excluded from stored content-token arrays. Check both
proposals are actual processed greedy argmaxes under their corresponding models. Reject suppressed,
out-of-vocabulary, non-EOS special/control or nonfinite candidates/logits; never fall back.

Exact inherited tie: choose theta0's proposal if `S_boundary(c0)-S_boundary(c2)>=-1e-12`, otherwise
A2's proposal. EOS wins → emit EOS, immediately terminate with unchanged content prefix and
terminated=eos. Content wins → emit exactly that content token under A2, consume it under A2,
resume ordinary A2, never trigger again. No theta0 handoff/G2 or copied cache.
Both models EOS → normal no-trigger termination. Both equal content → continue pre-trigger lockstep.
Cap remains the existing200-decision budget, not an EOS action. At the last allowed decision a
content win exhausts the budget and ends cap; EOS win ends EOS. No scoring/guard after exhaustion.

## Helper and integration boundary

Additive `csasr.inference_cf.eos_boundary` only: action_mode, boundary_scores, score_boundary,
execute_boundary. It is not a scientific runner. score_boundary uses one fresh cached.Branch per
model, prompt once then common prefix token-by-token, current-query logits only. It verifies common
prefix greediness, owned state hashes, contiguous positions and identical generation/tokenizer
settings. Two independent resident models; no cross-model decoder KV. Encoder share remains detached
and frozen. execute_boundary reuses original owned_clamp forcing one selected action; its EOS handling
already terminates before further continuation. No repeated detector exists in this execution helper.
Future runner must call the detector once logically, branch by live action types, then never invoke
another guard after commitment. Legacy metadata aliases A4/b4 for ordinary G1 identify actual A2 state.

The four known rows remain audit examples only. Historical panel status BLOCKED remains provenance,
not the current revision's authorization. EOS_FIRST/empty-content invalidity is superseded **only**
for the legal first-action EOS_BOUNDARY mode; no further fallback is allowed.

## Unchanged contract and additional diagnostics

All100 IDs/order/dialogues/source fingerprints, seven partitions/hashes, original A2 trainables/
teacher/loss/two AdamW updates/precision/reset, all100 reconstruction barrier, reference firewall,
TTA1 safety, ZH rescue>=5, benefit retention and PIER/MER/mixed guards, breadth/concentration,
bootstrap/LODO, terminal labels and precedence are exactly revision0. Current helper permits both
EOS orientations but does not assume outcomes of the four examples. Expected trigger set remains
all14 A2_DELTA; ten historical content/content sites use unchanged G1, four EOS-boundary sites use R1.
No hard-coded row-specific runtime rule. No reference or historical outcome informs boundary scores.

Before references, report EOS_BOUNDARY count and orientation (theta0-EOS/A2-EOS), winner EOS/content
counts, resulting termination and content length, signed length differences G−A2 and G−B0 per boundary
row. Preserve general trigger/opportunity logging. Boundary decision seal records both action IDs,
four log probabilities, H1, score/winner/margin, prefix, model ownership and selected action.
Only after complete100 outputs/logs are sealed **and pushed**, plus primary independent audit PASS,
report boundary-row POI/ZH/mixed error deltas G−A2 and whether EOS choices create new severe truncation.
The existing global severe-truncation safety gate is unchanged; no new diagnostic threshold/label.

## Tests and audit gates

Focused CPU tests cover content G1 identity, both EOS orientations, both EOS no-trigger, invalid action,
H1/no rollout, legal EOS probability, same processed logits/full normalization, weights/tie, EOS stop,
content commit/release, one event/no detector after commitment, independent owner caches/stale rejection,
no runtime row IDs, four known examples, exact panel/config safety/outcome/firewall inheritance.
Run existing TTA0/TTA-funnel/PATH1/PATH2/PATH3 tests and new PATH4-R1 tests on CPU only.

`experiments/inference_cf_p2path4_r1_contract_audit.py` is an independent standard-library freeze auditor:
no primary decision/helper imports, own numerical H1/tie fixtures, source/AST and historical-scope checks,
panel/partition/four-blocker reconstruction from sealed tokens, original config comparison, no PATH4
scientific result tree. It records **PASS_TO_P2_PATH4_R1** only after contract tests pass.
This is the new contract-level pre-run audit, superseding the impossible revision0 pre-gate.
Claude must ALSO complete independent runnable-manifest/environment/reconstruction/cache/code review
with PASS_TO_P2_PATH4_R1 before sbatch. A passed freeze audit does not authorize an unreviewed runner.
The post-outcome gate remains `P2_PATH4_AUDIT: PASS`; its auditor cannot import primary decision logic.

Later scientific scope unchanged: one job,100 rows,200 updates, max201 backward calls including live
audit, target<15min, hard30min. Ordinary regression comparator runs inside that allocation; no extra
job/objective/update. No A4/A3/G2, repeated guard, tuning, full300/P3/fresh/transfer data. No GPU in Codex.
Freeze audit and amendment/test/source hashes must be committed/pushed; clean local=remote.
