# SRD2-G0 implementation design — pending, not an experiment runner

This document specifies the smallest future implementation of the config/spec. This Codex freeze
adds a CPU identity inventory and contract tests only. No scientific runner, Slurm script,
model forward, lexical evaluator execution or old-experiment modification is implemented here.

## Reuse verified interfaces

| Component | Existing source/interface | Constraint |
|---|---|---|
| Population | `inference_cf_srd2_g0_inventory.build`, `identity_projection`, `order_key` | Metadata-only; already executed CPU |
| Audio/features | `csasr.models.whisper.load_audio`; R2 `_features` | Same mono16k waveform and heard/padding convention |
| Original R2 replay | `inference_cf_p0_r2.full_replay` | Full causal replay with attention, no cache |
| Attention/local LID | `core_r2.max_attention_window`; R2 `native_lid` | Frozen10 heads, 50frames,100-language probabilities |
| Gate math | `core_r2.local_support`, `conflict_from_logits`, `prefix_utf8_complete` | RAW script mass; unchanged partition |
| Cached queries | `inference_cf_cached.Branch`, `_processed`; `core_p1.processed_argmax` | Own pristine cache; explicit historical argmax ties |
| Site/readout | `DecoderPostCrossAttnRecorder`, `ReadoutDirection`, `readout_direction` | L16 native DG-02, zero-probe scratch autograd |
| Pulse | `p2r.solve_scale`, `scaled_direction`; `loc0_sites.pulse_action` and DG-02 intervention hook | Desired realized chord, not coefficient times gate |
| Storage | `inference_cf.core.atomic_json`, `digest`, `file_hash`; `utils.provenance`, `lss.specfreeze` | Immutable manifests, environment/source/config pins |
| Post-seal mapping | R2 `unit_to_token_positions`; canonical normalize/segment/align; P2-R `target_set` | Evaluator process only; never import from runner |

R2's runner also computes Ecf/Russian/donor controls; none supplies g_old and none is repeated here.
P2-R's orchestration consumes an evaluator-built oracle plan; **do not reuse that orchestration**.
Reuse its solver/branch only. P2-DIR's evaluator-gradient fields are not runtime inputs. S1/R0
regions/masks are not involved. `cached_greedy` uses `topk` for its selected ID; use the explicit
existing processed-argmax helper in the new canonical loop, as frozen in the config, rather than
assuming its tie order. No alternate generation processors or prompts.

## Minimal isolated future components

- `src/csasr/inference_cf/srd2_g0.py`: allowlisted runtime projection; dense query identity;
  deterministic within-utterance gate permutation; reference-free compatibility/dose ledgers.
- `experiments/inference_cf_srd2_g0.py`: prepare/manifest/capture/seal-a/permutation/pulses/seal-b.
  Imports mechanics above, never reference mapping or lexical target constructors.
- `experiments/inference_cf_srd2_g0_evaluate.py`: post-seal mapping and frozen metric/predicate
  implementation. No executable inference or direction construction.
- `experiments/inference_cf_srd2_g0_audit.py`: separate pre/a/primary/full paths. No import of
  primary gate-math wrapper, shuffle implementation, evaluator decision or candidate source code
  as a shortcut. Reads JSON config; independently recomputes equations and predicates.
- `slurm/inference_cf_srd2_g0.sbatch`: only PHASE=capture|pulses, each walltime03:00:00, one H100
  3g.40gb-compatible resource. No automatic job chaining or rerun/subset-on-timeout.

These are required future interfaces, not currently available execution commands. Do not build a
generalized multi-model framework, additional controls or a full-decoding method.

## Job A algorithm

1. Verify clean committed implementation, freeze/config/population/role/audio/model/source pins
   and pre-run audit. Prepare runtime JSON with exactly the four allowlisted identity/audio keys.
   Keep dialogue/roster/exclusion/evaluation metadata outside the runner input.
2. Per utterance, load/check waveform once; encode original audio once. Cached clean B0 greedily
   generates and records native logits at every actual query, explicit EOS/cap, content IDs,
   prefix hashes and cache positions. No steering hook or readout autograd exists in capture.
3. Replay prompt+fixed B0 content under the original R2 no-cache path with frozen heads; also
   record native DG-02 residuals, without edits. Slice current causal queries only. Retain the
   complete source values needed for independent reconstruction of windows/gates.
4. Recompute native null once per job; crop/pad original waveform using exact original localizer.
   Cache native LID only for identical crop/model/audio/preprocess keys, within utterance. Read raw
   float32 script masses from full-replay logits, combine with native null-adjusted E. Save
   components, raw selected crop hashes, fallback status, every query even when g=0.
5. Compare clean cached/full paths at all structural queries; run fixed prefix-only causal probes
   (first/median/last eligible t). Gate always retains full-replay attention/masses. Save the whole
   comparison distribution, not just passing queries. All config guards must pass globally.
6. Save elapsed/VRAM/unique-LID counts and complete inventory. Seal and push. Independent A audit;
   CPU freeze/shuffle/seal/push; complete-query recost and global authorization. No lexical access.

## Job B algorithm

First reproduce the unchanged historical apparatus using the P2-DIR audited30-query set, safely
projected to ID/t/prefix/source fields; the old set has zero membership in the new400 outcome
population. Reproduce saved clean logits/site, D2 direction and pulse/zero identities using old
frozen numerical contracts. This is engineering acceptance inside the same job, not an added
scientific comparison or an outcome-selected subset.

For each new utterance, rebuild the same original encoder output and clean cache. At each frozen
query verify its pristine output against Job A. At structural queries invoke the unchanged D2
scratch provider **once**, before baseline advancement. Record its native gradient, CPU tangent,
normalization evidence, direction hash, scratch-clean identity and cache/parameter fingerprints.
For B1/B2/B3 solve the predetermined desired chord on the exact clean site once (max8 emulations);
no solver call for zero target. Record pristine native q,u,r; preview the frozen DG-02 consumption
q+(u+(proposed_r-r)) on the native device. A failed consumed-energy or .005 norm-ratio guard is an
explicit numerical no-edit, preserving its planned dose. Reuse the solved scale through the
existing action interface with a hash-verified cached-solver callback; no second solve, coefficient
adjustment or different dose. Verify actual FFN input against the preview bitwise. For no-edit arms
reuse exact B0 without an unnecessary hook. Use independent pre-step scratch/crop restoration for each pulse, record the actually
consumed FFN input, lossless raw logits and processed top1. Check exact no-op identity and fresh
restored baseline after arms. Advance with the sealed **baseline token**, never the arm's token.
No continuation, beam expansion, teacher-forcing reference or optimized dose.

Keep the B3 donor assignment immutable even if a D2 direction is invalid or a chord unreachable.
The complete matrix includes recorded no-ops for every structural fallback. Never drop a row to
improve energy matching. Per-query lossless data include before/consumed native BF16 bits, proposed
repair, scale/emulation ledger, direction/gradient, raw full-vocabulary logits and processor hash.

## Schemas and sealing

Manifest: resolved config/hash; freeze/implementation Git commits/tree/dirty state; environment;
model/tokenizer/preprocessor/generation-config bytes and revisions; data/audio/runtime panel
hashes; partition/head/suppression pins; source hashes; job/status/time/VRAM; opened-path log.
Use project provenance/specfreeze utilities, with schema-versioned additions; never write into a
historical result directory. Existing attempts are immutable; no resume against changed pins.

Gate JSON: UID,t,absolute-query,prefix/native-input hashes, expected baseline action, actual EOS/cap,
structural status, full/cached comparison, window sample/frame boundaries/attention mass, native
100-way LID and null hashes, PE/PM/E/R/g, provider fallbacks and timings. NPZ: lossless distributions,
heard attention source, native site/probe evidence. Avoid JSON NaN/Infinity; statuses are explicit.

Permutation JSON: eligible recipient/donor t pairs, each original/permuted g and their bit hashes,
multiset hash, identity/constant/singleton status, seed/tag, gate-seal/config hashes. No strata.

Pulse JSON: frozen query key/direction hash, shared D2 evidence, four-arm target/realized chord and
squared energy, solver/no-edit reason, native consumption/restore identity, full-vocabulary blob
hashes/top1/cache fingerprints/timing. NPZ stores BF16 raw logits losslessly, not TopK truncation.
Authorization B: only the eight global fields listed in config; no IDs, gate ranking or correctness.

Each seal lists every shard/archive SHA, persistent immutable storage location, schema/counts,
source/config/population hashes and status. Commit/push seals and complete artifact digest manifests;
verify remote commit contains them. Preserve large raw archives on durable project storage with
content-addressed hashes (use existing Git/LFS policy where applicable, never omit raw data).
References stay locked until PRIMARY verifies every raw archive against the remotely committed
seal. The remote manifest is an immutable hash seal, not a claim that untracked scratch data are
permanently archived. Raw-data availability is an audit requirement.

## Tests required from Claude before PASS_TO_SRD2_G0

Add focused CPU tests for raw-R2 versus processed-D2 semantics, known partition/head pins, query
index/EOS/cap/UTF8 boundaries, causal no-future replay, source/cache ownership, exact zero,
e*g solver targets, unreachable-no-edit without a floor, shared direction, permutation multiset
including zeros/singletons, no reroll, schema/allowlist/label rejection, seal/authorization barriers,
bitwise restoration, lossless logit reconstruction and all predicate boundaries. Future tiny-model
CPU engineering tests are permitted in implementation; no such forward is run in this freeze.
Independently test evaluator mapping/collisions/target semantics/long-audio accounting and label
precedence with synthetic values. Auditor must not call the primary evaluator's decision function.

The current CPU contract tests cover frozen artifacts/roster/pins/predicates and pure historical
numerical functions only. Passing them does not confer the independent GPU pre-run audit.
