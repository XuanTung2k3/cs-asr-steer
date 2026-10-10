# R0 — implementation handoff, design only

Read `R0_REGION_VECTOR_SPEC.md`, config and `R0_REFERENCE_FIREWALL.md` together. No
runner/provider/outcomes implemented in this freeze. Historical primitives remain intact.
Exact numeric rules live in the spec/config; no change after model outcomes.

## Reuse verified interfaces

| Interface | Reuse / boundary |
|---|---|
| `models.whisper.load_audio`, `batch_model_inputs`, `load_whisper` | Source-compatible float32 waveform, mono/resampling, normal padded/truncated features and attention mask; explicit full-vs-heard sample coordinates |
| `experiments.inference_cf_p0_r2.native_lid` / `_features` | Exact100-token softmax, single SOT, independently30s-padded crop; classifier is new, LocalSupport gate unchanged and unused |
| `experiments.inference_cf_p0_r2.full_replay` | Causal full-prefix replay and frozen alignment-head aggregation; wrap one call with passive four-layer recorder rather than four replays |
| `lss.sites.DecoderPostCrossAttnRecorder` | Accepts arbitrary observed layer indices, states/q/u_source, keep_attention; no intervention hook needed; eval guard and consumed-r site test |
| `inference_cf.core_r2.prefix_utf8_complete`, `tokenizer_partition` | Query eligibility and diagnostic script partition only; do not let output script choose primary groups |
| `inference_cf.core_p1.direction` | Original prompt direction comparator with exact epsilon/fallback; no edits |
| `inference_cf.unique.moments`, `array_hash`, guard concepts | Float64 moments/serialization/orthogonality/sign evidence; **do not call fit_fold unchanged:** its rank32/min_count32/dialogue calibration is not R0 |
| `experiments.inference_cf_st_loc0.runtime_projection`, passive Engine recorder pattern, `loc0_sites.MultiSiteRecorder` | Projection and tested native r capture patterns; reuse cross-site primitive only; do not copy pulse/grid/calibration loops |
| `inference_cf.core.digest/file_hash/atomic_json`, `utils.provenance`, `lss.specfreeze` | Canonical JSON and versioned manifest/seal/environment/source/model pins |
| Historical dialogue-cluster bootstrap pattern | Independent adaptation to signed cosine endpoints, fixed Bonferroni8 family; no evaluator margin or steering endpoints |
| `nat5h.aligners.run_existing_ctc`, `data.ctc_alignment.align_units` | Inspect timing semantics only; **never invoke** or regenerate alignments in R0 |

Current source hashes are pinned in the panel/config. Historical model files pinned from
PATH5 include weights/tokenizer/preprocessor/generation config. Verify installed environment
and actual callable signatures before implementation. Source identity tests are pre-freeze
pins; later additive implementation must preserve inherited source pins or record an
explicit mechanical provenance revision before outcomes, never silently bypass mismatch.

## Proposed isolated additions, future Claude work

* `src/csasr/inference_cf/r0_regions.py`: pure grid/probability classification, sample-vote
  interval sweep, frame integration, common attention mapping, region controls. Provider
  has no reference/evaluator loader. Accept config, not row-specific exceptions.
* `src/csasr/inference_cf/r0_unique.py`: per-utterance small-rank thin-SVD constructor;
  raw state moments, centered **diagnostic** participation ratio, pre-winner rank selection,
  no retry after selected-rank winner guard. Status with all guard evidence; no substitute.
* `experiments/inference_cf_r0.py`: prepare/manifest/run/seal only. Project runtime inputs
  before file opens. Reuse exact baseline IDs, encoder once, M/E paired same-prefix replays,
  all layers in each replay. Per-row CPU fitting20 controls+2 perturbations, bounded RAM.
* `experiments/inference_cf_r0_analyze.py`: reference-free primary summary versus a strictly
  separate post-seal CPU oracle phase. The first mode cannot open oracle paths. Freeze
  oracle checksum, coverage, masks, vector guard/metric evidence and complete nested sets.
* `experiments/inference_cf_r0_audit.py`: independent numerical/region/decision implementation,
  no imports from primary analysis/decision. Preflight, PRIMARY and FULL modes.
* `slurm/inference_cf_r0.sbatch`: one H100/MIG allocation, hard3h, thread/memory bounds,
  manifest/pre-run PASS prerequisite; no automatic restart or arm/population reduction.

Names are an implementation plan, not existing symbols. Do not build a second Whisper
decoder, duplicate alignment/LID or import TTA training just to read historical B0.
Use sealed canonical B0 tokens; future R1 generation/cache application remains out of scope.

## CPU preflight and smoke contract

This design checked300/300 source audio and baseline bytes,300 unique IDs,20×15 dialogue
counts, canonical order, all requested historical files/interfaces, generation100 IDs and
ten alignment heads. Existing timing schema/UID/validity/provider metadata cover300 rows;
no boundary values, language memberships or reference surfaces loaded. Sidecar stale
data-hash field is explicitly historical; current source matches immutable P0-R2 manifest.

All originals read from disk in full before append-only canonical updates; tests assert
prior content remains intact. No model weights loaded, GPU invoked or R0 outcomes computed.
New focused tests check frozen manifest/population/source identity, grid/source geometry,
layer/head/threshold contracts, small-rank mathematical attainability/effective-rank
abstention, deterministic control bounds and native DG-02 causal replay on a tiny random
CPU Whisper. No test runs pretrained science or gold timing analysis.

Before production, implementation tests must additionally exercise:

1. EN/M/U classification including low-Q/silence/short windows; exact100 probabilities.
2. Tail/short/grid/crop bounds, full-waveform vs heard mask; vote conflicts/uncertainty.
3. Exact query indexing, t0/EOS/special/UTF8 exclusions, normalized real attention mass,
   no future-token dependence, all four layers share one membership; q+u native r.
4. Thin-SVD vs explicit moment reference; rank selection, repeated/correlated samples,
   gaps/ties/sign degeneracy; fail without fallback, complete invalid records.
5. Shuffle label/sample counts and seeds, erosion/dilation conflicts, independent refit.
6. No gold/reference-provider inputs/imports, reference mutation no-op, weights/grads/
   hooks unchanged, array/model/config/feature/prefix caches exact-hash keyed.
7. Nested quantitative gates, empty denominator/strata, shared dialogue bootstrap draws,
   Bonferroni8, no post-outcome layer/rank/shuffle optimization, PARTIAL does not advance.
8. Oracle source projection/filtering/coordinate errors, duplicate/conflicting intervals,
   same sealed states/queries/attention; no reference teacher forcing or new aligner.

## Cost and execution order

Exact CPU grid count10387 windows over5269.3300625s original audio. Historical R2
1428s with11855 real LID plus8750 controls and B/E replays gives order-of-magnitude
15–40min prediction; ST-LOC0 passive80-row extraction71s supports cheap four-layer
capture. Thin row-SVD cost is proportional to query-count²×1280, not1280³;20 shuffles
must not call dense moment eigendecomposition. Future CPU synthetic maximum-size timing
plus first canonical-row integrity/cost smoke must conservatively fit3h for all300.
No cost is inferred from uninspected R0 agreement outcomes. If it cannot fit, STOP for
explicit pre-outcome compute redesign. Do not reduce population, layers or controls.

Design-session reference-free CPU benchmark: synthetic100/99 group rows×1280, two raw
thin SVDs plus two centered diagnostic SVDs, ten repetitions, single BLAS thread:
0.02547s per layer/control cell. Conservative all300×4×24 constructions (primary,
20 shuffles, two perturbations, script diagnostic) projects ~734s CPU algebra, before
serialization/mapping overhead. No real R0 states, memberships or outcomes were used.
Combined with historical LID cost this supports the provisional15–40min estimate;
it does not replace implementation preflight or the hard3h ceiling.

Claude order:

1. Fetch/check newest local work, read current authority plus frozen R0 docs/config/panel.
2. Implement isolated reference-free provider/constructor/runner and independent auditor;
   CPU tests, synthetic throughput preflight, review/commit/push. Historical sources intact.
3. Verify model/audio/baseline/oracle structural fingerprints without opening boundaries;
   freeze manifest/environment, `PASS_TO_R0`, commit/push before one scientific job.
4. Inside allocation run limited integrity/cost smoke; full300 if scope/time valid; persist
   all states/attention/primary vectors/controls/reasons. No evaluator access.
5. Seal complete primary arrays/manifest, push, verify remote, independent PRIMARY PASS.
6. CPU oracle/evaluator ONLY now: filter allowed IDs, construct proxy masks/vectors using
   sealed states, compute frozen measures/gates/labels. Independent FULL PASS, push report.
7. Stop. READY recommends a separately frozen R1; PARTIAL or failure authorizes no extension.

No confirmation/test/P3/transfer, no new learned model, no controller, no activation pulse,
no direction optimization, no gold omission repair. Historical conclusions unchanged.
