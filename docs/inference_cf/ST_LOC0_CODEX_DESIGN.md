# ST-LOC0 — implementation and independent audit design

This file plans future execution of `ST_LOC0_SPEC.md` / `st_loc0.json`.
**Freeze only:** no scientific runner, optimizer, GPU job or outcome report is added here.
The exact self-attention adapter remains implementation work; its residual geometry is verified
on the real Whisper layer class with synthetic CPU inputs. Core v6/DG-02 remains unchanged.

## 1. Audited reuse map

| Component | Exact existing implementation/artifact | ST-LOC0 use |
|---|---|---|
| V1 / D0 | `core_p1.direction`, `directions.OldDirection` | Same-prefix E−M; identical precision/epsilon/fallback at every site |
| V2 / D1 | `unique.fit_fold`, `fold_members`, `array_hash`; `directions.UniqueDirection` | Fixed clean membership, site-specific float64 crossfits; sealed runtime vectors |
| Matched query replay | `inference_cf_p2r.DiagBranch`, P2-DIR `extract_utterance` / `exp1_utterance` | B0M_L16 content prefix; crop/restore own pre-query cache |
| Cross residual | `lss.sites.DecoderPostCrossAttnRecorder` / `DecoderPostCrossAttnInterventionHook` | q/u_source/r; original DG-02 L16 and same site L24 |
| Norm repair | `models.hooks.apply_steering` | Native bf16, epsilon 1e-6, no depth rescale |
| Fixed energy | `inference_cf_p2r.solve_scale`, `pulse_hook` | Reuse solver; cross hook unchanged; self-site thin adapter |
| Historical states | `p2dir/extract_run1/states.npz` / per-row arrays | Original L16 HB/HE, verify passive replay; no redundant old extraction |
| Historical vectors/outcomes | `p2dir/folds_run1`, `exp1_run1/directions_sealed.json` | All 20 D1 folds and 180 D0/D2 vectors; packed bf16 old pulse logits |
| Evaluator | `inference_cf_p2dir_analyze.logit_metrics`, `draw_weights`, `boot_stat` | Fixed target/competitor, dialogue means, same RNG; adjust family size to 16 |
| Independent auditor | `inference_cf_p2dir_audit` patterns | Independently derive math/metrics/decision, no primary decision import |
| Provenance | `csasr.utils.provenance` / `csasr.lss.specfreeze`, inference_cf manifests | Resolved config/git/env/model/data/state/cache hashes and immutable phase seals |

Historical original model is `/mnt/data/tungnx/whisper-large-v3`; model/tokenizer/config hashes
are pinned from Exp-1 in config. Historical run source hashes and current code hashes are separate
provenance: retain both. Changing a hash never makes a scientifically different implementation
compatible; independent audit must explain any mechanical adapter addition before running.

A5 provenance is explicitly different: English span states versus preceding states, reference
prefixes/top-64 intermediate in the production builder. Read `basis_a5_unique_shared.py` and its
manifest only for history; do not use its Add-Unique copy as the clean D1 provider.

## 2. Three input boundaries

1. **Calibration builder:** read the sealed 180-key four-field construction projection and
   site-specific unedited B states; use only A/B membership / dialogue exclusion. No target tokens,
   previous pulse outcomes, CTC/evaluator gradients, timing, utility or reference margin fields.
2. **Pulse runner:** load `runtime_queries`, baseline-generated content tokens/audio fingerprints,
   prompts/settings and sealed vectors. Do not import evaluator loader or read construction labels.
   Match original same-audio/prefix semantics, with no future content query used as a feature.
3. **Evaluator:** only after full output seal is committed/pushed; read original P2-RJ targets,
   fixed competitor and strata. The oracle exists only here.

The panel freezes both approved membership and label-free runtime projections; do not hand its
entire JSON object to a provider. Test schema allowlists and disallowed-import/argument rejection.
No scientific outcome is needed to build these projections.

## 3. Exact self-attention residual adapter

Inspect/pin installed `WhisperDecoderLayer.forward` before implementation. In eval mode:
self_attn_layer_norm pre-hook captures h_in; self_attn forward hook captures output[0]=u_self.
Compute q=h_in+u_self IN NATIVE DTYPE, then run historical repair on only the absolute current
query. Return `(u_self+(q_repaired-q), *original_tuple_tail)`. Leave every other sequence position
unchanged, and return the original output unchanged for zero/invalid dose. The cross-LN pre-hook
records ACTUALLY CONSUMED q. Same dtype, shape and tuple/cache/attention outputs are mandatory.

An encoder_attn_layer_norm argument-only edit is forbidden: the layer already stored q as its
residual, so that edit would change only the attention calculation. The residual-aware adapter
changes both the source input and bypass. Native recomposition rounding is recorded and checked
by the same consumed/proposed tolerance as historical DG-02. Do not claim exact repaired q without
measuring the consumed tensor. Passive self recorder returns outputs unchanged.

Cross uses existing DG-02 hook, audit final-layer-LN input r. Shapes `(B,T,1280)`; logical query
4+t−1; no forced-prefix edits. A full-prefix scratch pass and cached single-query pass must yield
the same semantic site, and both must have zero-dose bitwise identity. No model parameter changes.

## 4. Frozen extraction, fitting, and calibration seal

Reuse original L16 cross matrices and clean fold files after hash/replay checks. Passive B/E
observation can capture all four sites during one paired replay; verify that multi-recorder
observation leaves original logits/states unchanged. Keep native bf16 arrays losslessly as
float32 or packed bf16. Reuse same prefix and encoder cache for paired conditions as old P2-DIR.

Fit the other THREE sites independently (60 new fold fits), same 120 clean construction positions,
whole-dialogue exclusion, exact top-32 SVD/guards. Sort memberships deterministically. Persist
moment hashes and reusable state matrices/eigenvalues/principal values/score vector/chosen index/sign/unit tests/vector hashes and
invalid reasons, not merely the final vector. Original 20 fits refit only for provenance audit;
the freeze's CPU-only check reproduced their vectors exactly (all 20 maxabs 0).

Hash ALL per-query V1 directions before pulses. Four random controls are predetermined in config;
verify their array hashes. D2 reuses independently verified historical float32 vectors. Seal all
80 site/fold records and query vectors before any pulse outcomes. Commit/push immutable calibration
seal; if push/identity/audit fails, STOP. Future single allocation can pause for this CPU/push phase;
network/sealing overhead must be included in the 3h estimate. No second scientific job authorized.

## 5. Historical barrier and matrix scheduling

FIRST replay NONE/D0+/D1+/D2 at L16 DG-02 on all180, check exact historical outputs and source/
vector identities. These are already part of the final matrix, not an extra sweep. Barrier failure
aborts before other pulses. Historical evaluator scalars are rechecked after the full seal; do not
open references in the pulse process just to reproduce old counts.

Then complete the other primary arms and four random controls. For each query retain an untouched
pre-query cache; cloned/cropped replay for each arm uses only that cache and the original input token.
Do not use a steered output as the next prefix. Assert cache-position lineage; hash/check all prefix
cache tensors; truncate/rebuild under original weights if needed using the historical path. Fresh
NONE replay after each group must be bitwise equal. Remove all hooks on success/exception, assert
none remain. ZERO at all four sites is bitwise baseline. No autograd except optional independent
historical D2 spot verification; no gradients construct V1/V2 or pulse strength.

Record all3780 pulse cells, including predetermined mathematical no-edit reasons. No missing cell,
new strength/rank or substitute site. Energy failures must be identified from directions/solver/site
states, never from reference outcomes. Eligibility is >=57 valid rows and >=12 valid dialogues in
EACH stratum; all invalid cells stay zero-effect in denominators. This prevents selective reporting
of attainable/favorable rows. Below-threshold arms remain in the multiplicity family of16.

Store compact metadata plus lossless bf16 next-token logits and native before/proposed/consumed
states for this bounded diagnostic panel. Estimated packed logits ~0.41GB uncompressed for3960
full-vocabulary distributions; avoid duplicate NONE arrays and float64 logit archives. Save runtime,
model calls, solver calls, peak allocated/reserved VRAM and invalid counts per arm/site. Large arrays
stay in result artifacts, not in documentation/report JSON. Seal file hashes and remote identity
before opening evaluator data.

## 6. Post-seal analysis, labels, and independent audit

Reuse fixed historical margin `logsumexp(Y_ref)-z[competitor]`, ranks and script mass from original
partition. Every arm gets all three strata. Dialogue-macro Δmargin is both materiality estimand and
bootstrap statistic; position average is secondary. Bootstrap B10000 / seed240924, same shared
sorted dialogue draws, Bonferroni family16 adjusted quantiles `.0015625/.9984375`. No outcome-fitted
threshold, no per-arm bootstrap seed, no rank/group refit in bootstrap. Safety is descriptive.

Primary label logic is the disjoint spec §11: INVALID, no eligible adjusted-positive arm, positive
but no fully qualifying arm, or SITE_FEASIBLE. Feasible gates: macro >=.50 nat, adjusted lower>0,
>=3 reference-correcting flips across>=3 dialogues and dose/integrity PASS. Winner order is frozen
in config. Independent auditor must implement this logic separately and verify evaluator scalars
within1e-8. D2/random cannot qualify or inflate the candidate-only oracle. Report correct-state
damage even for winner and oracle; no safe/deployable claim.

Pre-run **PASS_TO_ST_LOC0** and post-run **ST_LOC0_AUDIT: PASS** are required. Auditor may reuse
basic hashing/site primitives, but must not import primary fitting/analysis/decision functions
for its numerical audit (independently recompute moments, SVD, energy, metric/interval/label logic).
Audit trainable/frozen state, source provenance, reference allowlists, exact matrices, cache lineage,
manifest seals, all predicates and next-step scope. No continuation on failed audit.

## 7. Focused tests and execution order for Claude

CPU freeze tests pin population hashes/prefixes/folds and full grid/random hashes; refit
all20 historical D1 vectors and reproduce180 D0 vectors from sealed states; synthetic real-Whisper
self-site route and zero-edit identity. Existing P2-DIR/P2-R/DG-02 tests protect historical providers,
solver, packed logits, norm repair and exact cross-site plumbing.

Future implementation adds only focused tests for production cached/full-prefix self adapter,
allsite zero/restore identity, prefix/cache/hash integrity, invalid no-edit/arm eligibility,
runner allowlist/firewall, matrix completeness, adjusted shared dialogue bootstrap, exhaustive
label/winner ordering and independent audit agreement. No experiment outcome test or full runner
is implemented in this Codex freeze.

Claude order:

1. Verify local/remote freeze, all source/panel/calibration identities, historical provenance and
   model forward; preserve newer intentional work.
2. Implement thin site adapter + reuse-first builder/pulse/evaluator/auditor; CPU checks, inspect
   diff, commit/push. No GPU until independent PASS_TO_ST_LOC0 and full-matrix time estimate <=3h.
3. ONE sbatch allocation: passive paired capture → CPU folds/query directions → immutable calibration
   PUSH → historical barrier → complete frozen pulses. Stop on mismatch, no alternate configuration.
4. Immutable all-output manifest/hash PUSH; only then CPU post-seal evaluator/bootstrap/oracle and
   independent post-audit. Report every planned arm/invalid cell and unchanged historical conclusions.
5. Commit/push report/decision; verify clean local==remote. SITE_FEASIBLE recommends a separate LOC1
   freeze only; no optimizer, full decoding, new dose/direction or fresh data is authorized here.
