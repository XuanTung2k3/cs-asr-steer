# S1 implementation design and reproducibility review

Design-only freeze, no science runner implemented or model forward executed. Supporting inference_cf
study; no v6 METHOD_CONTRACT change. Numerical/decision authority is S1 spec/config. The panel copies
parent query/utterance order exactly and fingerprints 80 R0 primary records and 80 historical LAC
masked-logit archives (180 queries). Audio identity is full-file SHA plus historical first64KiB identity,
not a mismatched waveform/hash substitute.

## Reuse verified interfaces

- `csasr.models.whisper.load_whisper`, `load_audio`, `batch_model_inputs`: existing local model,
  soundfile float32/channel mean/resampling and native feature extractor/real attention mask.
  Load the exact `configs/model/whisper_large_v3.yaml`; require CUDA bf16/eager, abort on CPU fallback. Verify
  actual callable signatures and model/generation/processor file hashes during preparation.
- `experiments.inference_cf_cached.Branch.step` and `_processed`: owner KV lineage, prompt then
  single-token prefix steps, float32 logits, frozen alignment-head capture, exact suppression.
  Reuse only these primitives, not the steering/gate driver. No intervention hook is installed.
- `WhisperGenerationMixin.detect_language` from the pinned installed Transformers file: original
  encoder/SOT detection across all100 language IDs, original-only, task transcribe prompt. A stubbed
  `_retrieve_init_tokens` test must verify the prompt with no pretrained forward or model loading.
- `csasr.inference_cf.r0_regions.query_mapping`, `intervals_to_track`: primary heard EN intervals and
  mean-head normalization. The source's .5 real-mass guard is retained; S1 does not use assign_queries
  or unique fitting. Source R0 head arrays are not reused for different prefixes/replay numerics.
- `csasr.inference_cf.lexical_compatibility.hard_mask`, `waveform_model_inputs`: byte-safe positive
  zeros, heard bounds and whole masked acoustic preprocessing. These helpers have no references.
- `core_r2.tokenizer_partition`: unchanged script annotation for descriptive/correct-Mandarin
  protection in evaluator only. No script class restricts candidate generation.
- Canonical P2-RJ acceptable first-token sets and frozen evaluator; copy endpoints/definitions,
  do not invoke gold-aligned preparation from runtime. LAC masked archives are packed bf16 full
  vocab; unpack without changing values and process with the same original query suppression.
- `csasr.utils.provenance` / `csasr.lss.specfreeze`: resolved config, source/env/model/hash manifest;
  immutable atomic artifact creation, owner cache and open-file audit logs. No overwrite/resume
  across incompatible manifests; failed attempts preserved under new numbered directories.

## Proposed isolated components (future, not falsely marked present)

`src/csasr/inference_cf/s1_evidence.py`: pure runtime projection, stable top-K/union, interval
integration, target/off-target choice, immutable candidate rescore. No evaluation imports.
`experiments/acoustic_s1.py`: one command surface
`prepare | candidates | candidate-seal | acoustic | seal | evaluate | audit`, with
`--config configs/inference_cf/s1_acoustic_evidence.json --out results/inference_cf/s1/run1`.
`experiments/acoustic_s1_evaluate.py`: separate reference-only evaluation/decisions.
`experiments/acoustic_s1_audit.py`: independent prerun/candidates/primary/full implementation;
may reuse pure IO, never primary selection, scoring or decision functions.
`slurm/acoustic_s1.sbatch`: phase argument; two sequential jobs at most, 03:00:00 ceiling.
Dispatch evaluation lazily in an isolated process after verifying remote seal and PRIMARY audit.
Do not import evaluator at runner module import time. Existing historical files stay byte-identical.

## Frozen schemas and artifacts

Preparation `results/inference_cf/s1/plan_sealed.json`: runtime projection without evaluation keys,
source/model/tokenizer/suppression and audio/region hashes; exact180/80/20, query-prefix consistency,
installed AUTO prompt semantics and historical M/LAC compatibility. Commit/push before inference.
Manifest per phase includes clean git HEAD, env, resolved config, SLURM ID, model files, timing,
peak allocated/reserved VRAM, encoder/decoder/attention call counts and status.

Candidate `run1/candidates/NNN.json` per utterance: UID/audio/prefix hashes, detected language token,
queries with Top5/20 each source, ordered unions/IDs/origins/ranks, EOS/legal type, input IDs, cache
lineage; original M real attention, selected target/off-target bounds/status/energy/mass, R0 hashes.
`NNN.npz`: full branch raw packed-bf16 logits, processed float32 log-prob vectors, original M heads;
arrays index/dtype/shape/hash in JSON. No references. `candidates_sealed.json` recursively hashes
all candidate arrays/rows/config/manifest, candidate membership digest per query/budget; push first.
Region association uses original-only attention and may be sealed with candidates, never revised.

Acoustic `run1/acoustic/NNN.json` / `.npz`: fixed union digest, waveform/feature/encoder/cache hashes,
mask bounds/eligibility/outside bytes, full M masked raw logits/log-probs and candidate support/score
for target, off-target, historical LAC, permutation and null; tie-ranked fixed token IDs. Include
all expected queries even on region/control abstention. No regenerated candidates or gold analysis.
`output_seal.json` covers both phases and original seal/config/source hashes; immutable push and
remote ancestry check required. `primary_analysis.json` contains reference-free completeness,
coverage/status, candidate sizes/EOS/branches, region overlap/duration/energies and efficiency only.

Post-seal `analysis.json`, report, FULL audit: H/D pooled+macro/ranks/corrections/corruptions,
bootstrap family/quantiles/draw validity, all strata and branch/mask-control comparisons; structural
unreachable English limitations from historical evaluator; exact terminal decision. These outputs
are NOT created in the design session. No large tensors committed except required sealed distributions.

## Required implementation tests and independent audit

Tests before PASS_TO_S1: exact panel/query hashes; label-free projection; M replay bitwise180;
M/E/AUTO same content prefix/task with own cache; native AUTO original detection/all language IDs;
no masked detection; stable ties/dedup and EOS legality; candidate union immutability; suppression/
full log-softmax; attention heads/partial frame integration; heard horizon; absent-region abstention;
one deterministic target/off-target duration/energy guard; outside byte identity; full masked encoder;
no original decoder cache transplant; S identity/lambda; permutation determinism; full LAC prefix/
raw distribution equality; no autograd/weight changes/hooks; no gold files; complete query coverage;
independent count/label precedence; dialogue bootstrap empty-draw and multiplicity behavior.

Independent audit derives candidates from full original distributions, region bounds from primary
intervals/fresh attention, mask identities from waveform bytes, scores from raw masked logits,
and post-seal reference hits/counts/labels without importing primary analysis/decision. Any integrity
failure stops. Expected region/control abstention is an observed feasibility limitation, not invalidity.
No PASS is claimed by design CPU tests alone; executable preaudit remains mandatory.
