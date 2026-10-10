# SRC-CF0-P — architecture and implementation obligations

Design freeze only. No science runner, evaluator, auditor or Slurm launcher is implemented by
this ticket. Two real additions are CPU-only `cf_pilot_contract.py` (pure numerical/aggregate
predicates) and `inference_cf_src_cf0_safety_inventory.py` (existing artifact census/bounds).
They load no model and perform no scientific intervention. Future executable code must be
reviewed against this specification and independently audited before any allocation.

## Verified reusable primitives

| Interface | Reuse and constraint |
|---|---|
| `s1_evidence.heard_attention`, `Integrals`, `select_target`, `select_offtarget` | Exact S1 association/crop ordering; no reranking/candidate dependency |
| `r0_regions` PRIMARY heard intervals | Reuse sealed `track_heard_intervals` only; no R0 oracle or full-replay heads |
| `lexical_compatibility.hard_mask`, `waveform_model_inputs` | Unchanged waveform hard-zero and feature preprocessing |
| `inference_cf_cached.Branch.step` | Cached prompt/content queries, independent audio-owned caches; `_processed` semantics unchanged |
| `DecoderPostCrossAttnRecorder`, `DecoderPostCrossAttnInterventionHook` | DG-02 q+cross output residual, native pre-FFN consumption; layers16/24 |
| `core_p1.direction` | Float64 contrast/epsilon guard concept; pilot paired clean/mask numerical helper has no prompt/gold arguments |
| `loc0_sites.pulse_action`, `cross_pulse_hook`, `Composite`, cache/hook guards | Owner-scoped one-query pulse, hook restoration; no post-self adapter |
| `inference_cf_p2r.solve_scale`, `scaled_direction`, `emulate_edit_norm`, `DiagBranch` | Matched native BF16 chord, crop/snapshot pristine cache, solver statuses |
| `models.hooks.apply_steering` | Frozen NormPreserve, no depth rescale; norm-rounding diagnostic only |
| `utils.provenance`, `lss.specfreeze` | Resolved manifest, artifact/file hashes, environment and git state |
| S1/ST-LOC0 independent auditors | Reuse generic hash/remote/open-file utilities; independently reconstruct scientific formulas |

S1 cached runner archived logits and attention, not DG-02 residuals. Residuals require new,
future passive same-prefix forwards. R0 states use a different full-replay path and cannot be
substituted. `Branch.step(capture_layer=...)` captures one layer; to obtain both in one pass,
use an owner-scoped multi-layer passive recorder context (`hook=recorder`, capture_layer=None),
then verify q+u equals the native tensor entering each FFN's final_layer_norm. Confirm the
actual installed Transformers implementation/source hash before running. Never hook block output.
The clean prefix must reproduce historical B0 logits; no model/generation change is tolerated.

## Proposed isolated implementation

- `experiments/inference_cf_src_cf0_pilot.py`: `prepare`, `manifest`, `construct`, `seal-a`,
  `pulse`, `seal-b`. Preparation writes safe runtime JSON; construction captures and seals
  directions/geometry; pulse accepts sealed directions plus global P-B authorization only.
- `experiments/inference_cf_src_cf0_pilot_evaluate.py`: separate `gate-a` (historical strata,
  no lexical target access), then `evaluate-b` after primary audit. No evaluator import in runner.
- `experiments/inference_cf_src_cf0_pilot_audit.py`: independent `pre`, `construction`,
  `primary`, `full`; does not import `cf_pilot_contract` decision/numerical helpers.
- `slurm/inference_cf_src_cf0_pilot.sbatch`: future PHASE=construct|pulse, one allocation per
  phase, <=3h, H100/MIG, project acl1 Python; requires pushed phase-specific audit/authorization.

These are required CLI interfaces, NOT currently runnable commands. Do not invoke a missing
script or pretend this freeze is a runnable audit. Existing S1 files remain unchanged.
`cf_pilot_contract` supplies synthetic CPU reproducibility/threshold tests and aggregate decision
predicates; it is not a direction inference provider or a Whisper experiment.

## Required output schemas under results/inference_cf/src_cf0_pilot/run1

`runtime.json`: exact schema allowlist,180 queries/80 utterances, prefix truncated to max query,
region records and source hashes only; separate artifact hashes for cropped mask bytes.
`manifest.json`: resolved config, source commit/file SHA256, environment/Transformers/CUDA,
model/tokenizer/preprocessor files, generation suppression/prompt/head metadata, dataset role,
panel/runtime hashes, seed, git cleanliness, status, Slurm identity, no-autograd declaration.
Record reuse provenance per tensor, never dimensions alone.

`construction/NNN.json` and `.npz`: query UID/t, input/audio/prefix hash, clean/target/off masks,
layer16/24 native residual bytes (packed BF16 int16), full original raw float32 logits,
repeat hashes, CPU float64 delta and float32 vector, norm/cosine/tangent evidence,
all sign/dose solver records including failures, random vector seed/hash and independent
owner/cache hashes. Maintain every original query; avoid leaking historical strata.
Target-only rows are captured and reported. Off missing means explicit paired abstention.

`direction_seal.json`: all file hashes, resolved config/source/runtime hashes, expected coverage,
remote source commit, references_used=false, stable seal hash, complete matrix counts.
`construction_audit.json`: independent numerical/identity PASS and evidence hashes.
`gate_a.json`: evaluator-only detailed layer/stratum counts (never a runner input).
`PB_authorization.json`: minimal global config/seal/audit hashes, pass boolean, qualified layer IDs,
time and self hash; qualified layers are global developmental decisions, no row labels.

`pulses/NNN.json` and `.npz`: all8 primary/8 random/8 off records, NONE/zero hashes,
requested/actual eta, scale/solver status, proposed/consumed native state, pristine cache hash,
full-vocabulary raw float32 logits or baseline reference for no-edits, restoration checks,
direction/mask/source hashes. Store raw logits once; processing independently reproducible
from sealed generation config. No references/strata/rank-to-gold fields. Zero identity can
store equality hashes instead of redundant full tensors. Target-only outputs remain separate
from paired-comparison cohort; no matched-cohort enlargement after evaluation.

`pulse_seal.json`, `primary_audit.json`: complete outputs and firewall/identity PASS, pushed
before lexical references. `evaluation.json`/`full_audit.json`/future pilot report: acceptable
first-token counts, paired cohorts, macro/bootstrap results, active safety rows, exact terminal
and selection; reference-only. Failed attempts go to distinct preserved directories.

## Minimal future model adapter documentation

A later backbone adapter would expose immutable audio encoding, prompt/content-prefix replay,
exact native residual capture, single-site repair and generation-valid logits. This ticket
implements no generalized speech-LLM API, backbone experiment or cross-model compatibility.

## CPU preflight and production forecast

Use `/home/tungnx/miniconda3/envs/acl1/bin/python`, PYTHONPATH=src:., OMP/MKL/OPENBLAS threads1.
Contract tests use synthetic arrays and existing sealed metadata only. Only tiny random-model historical regression fixtures are allowed; no pretrained model loading,
new scientific LID/encoder pass, masked inference or steering outcomes in Codex. Production source hashes
must be appended to a pushed resolved manifest after implementation; inherited pins cannot
silently change. Confirm throughput via reference-free engineering-only preflight during later
Claude implementation, not by inspecting lexical outcomes. If predicted >3h/phase, STOP;
no scientific matrix reduction. CPU FULL300 script is independently reproducible now and is
not a science runner or authorization for its proposed future GPU attention census.
