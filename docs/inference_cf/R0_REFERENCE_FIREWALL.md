# R0 reference firewall — frozen before outcomes

R0 is a reference-free provider plus a post-seal reference-derived timing evaluator.
It is not a gold-assisted inference method. Read `R0_REGION_VECTOR_SPEC.md` before coding.

## Allowed primary inputs

The executable projection of `R0_PANEL.json` is **rows only**: canonical_index,
utterance_id, dialogue_id, audio geometry/bytes/hash, and sealed generated baseline
content IDs/termination/hash. Global source/model/config pins may be verified, but the
runner must not open oracle_source.path or role reference columns. Decode source JSONs
are read through exact theta0 projection; A2/controller/evaluator fields must not be
passed onward. Runtime ID does not choose language membership or numerical rank.

Provider signatures take waveform,100-way native LID probabilities, baseline tokens,
frozen alignment-head attention and exact DG-02 states. No reference, gold_region,
target_token, POI, correctness, CTC boundary, ref/hyp alignment or oracle vector argument.
No A5/D1 artifact loading. No training, optimizer/backward, steering/intervention, TTA,
router or hidden gold prefix. Cached model-owned artifacts must have exact full input,
preprocessing/model/window/prefix/query/site/config hashes, never approximate matches.

The primary process may not import `r0_analyze` oracle code, CTC/reference/evaluator
loaders, role-manifest transcript columns, historical per-position correctness membership,
or historical region boundaries. Static code/contract inspection is allowed. The design
session verified only timing-source schema/IDs/validity/provider metadata and hashes;
it did not read gold boundaries or construct new R0 outcomes.

## Seal before any oracle access

Complete300-row coverage, all four layer states, audio/sample and baseline hashes,
native window probabilities/labels/abstentions, sample/frame masks, every eligible and
excluded decoder query, attention heads/raw mass/normalized frame vector, grouping code,
rank-candidate guard evidence, primary/perturbed/shuffled vectors and reasons, shuffle
seeds/offsets, prompt/script descriptive outputs, source/config/model/git/environment
manifest and output hashes must be immutable and pushed. Large matrices may remain
hashed external files; publish locators, shapes, dtypes, full hashes and accessible
archives for audit, not only scalar summaries. Confirm actual remote seal commit.

Independent PRIMARY audit PASS is required. Neither references already being exposed
nor an uncommitted seal allows bypass. A failed audit halts oracle access.

## Oracle-only phase

Separate CPU evaluator filters CTC to the300 frozen IDs before materializing reference
columns. It may open already-exposed role references only for reference-unit consistency
and English-omission diagnosis. Existing reference-derived MMS-FA timings are an oracle
proxy, not manual acoustic truth. Preserve current source bytes and sidecar discrepancy;
do not regenerate alignments or visit any other split. Gold intervals replace sample
mask membership ONLY; the sealed baseline states/query set/attention are immutable.

Use identical mapping and numerical policy; gold means orient only oracle vector, never
prediction. No oracle mask/vector/boundary/label enters a future provider, selector,
prompt comparator or numerical fallback. Final construction agreement is not ASR gain.

## Independent audit and leakage tests

Audit source/import dependencies and runtime file-open logs; provider argument allowlist;
mutation test changing references leaves all primary arrays byte-identical; synthetic
test proving replayed future decoder suffix cannot change earlier query; query-to-frame
mapping remains layer-invariant. Check all model parameters unchanged/frozen, gradients
None, no edit hooks, encoder output detached, no hidden oracle teacher forcing.

Auditor must independently reconstruct primary and oracle group masks, every rank choice
and abstention, SVD scores/sign, control generation, all bootstrap endpoints and nested
label precedence. It must not import primary decision implementation. Require
`PASS_TO_R0`, `R0_AUDIT: PASS (PRIMARY)`, then `R0_AUDIT: PASS (FULL)`.

Allowed: existing FULL300 D-dev-select and inference_cf history. Forbidden:
D-dev-confirm, D-test, router-calib as new role, P3, all transfer datasets. No new split,
no gold-assisted repair, no automatic R1. Core v6 and closed historical stages unchanged.
