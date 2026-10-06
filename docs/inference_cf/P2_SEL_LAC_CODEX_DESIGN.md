# P2-SEL-LAC — implementation handoff

Clean starting local==origin/cs-asr-steer-inf at `6421b0cae58e3e2836c8ff9eaeeb8e41668c0550`, branch
feature/inference-cf-steering, no local-only work or queued jobs. Contract: P2_SEL_LAC_SPEC.md and
configs/inference_cf/p2_sel_lac.json. Bounded development only; core METHOD_CONTRACT unchanged.
No LAC runner, candidate pair extraction or scientific outcome in design session. Historical E/T/XA
terminal stops/reports remain intact, P3 HELD.

## Actual implementation map and reuse

| Need | Reuse / implementation instruction |
|---|---|
| Input waveform | src/csasr/models/whisper.py:load_audio; float32 soundfile, mono mean, existing scipy resampling. Mask at16000Hz after loading, before feature extraction. |
| Waveform-array features | batch_model_inputs currently takes PATHS and returns actual feature-extractor attention_mask. Add only a tiny in-memory adapter in the new module, same feature_extractor list/kwargs/defaults/device/dtype. Validate bitwise against original path on all80 unmasked inputs before outcomes. Do not export/re-read WAV. |
| Encoder | bundle.model.model.encoder(input_features=...,attention_mask=...).last_hidden_state; BaseModelOutput wrapper, same old80/180 provider/model metadata. Masked encoder rebuilt, not frame-zeroed or swapped into unmasked cache. |
| Candidates/partition | core_r2.tokenizer_partition; V_E=embedded_ids37858, V_M=matrix_ids1667; hash pinned. Raw P2-DIR NONE bf16 unpack, sorted token-ID argmax. Suppression intersection verified EMPTY for both lists, including step0. No actual candidate IDs computed in design. |
| Raw baseline | p2dir exp1_run1/directions_sealed.json and *_logits.npz t{t}_none; unpack_bf16 lossless. XA0 independently confirmed180 live B sites/logits bitwise; reuse engineering evidence even though XA scientific FD invalid. No unmasked rerun. |
| Prefix | p2dir.BASELINE_RUN B0M_L16 tokens, CB=[50258,50260,50360,50364]; exact prompt+tokens[:t], matched absolute query. Never token t/reference/future continuation. |
| Masked replay | p2r.DiagBranch / cached.Branch.step, fresh empty self/cross cache using masked encoder. Prompt step then baseline tokens one-by-one, attention=True as original cached path. No source-scale/gradient hook or masked generation. |
| W* | p2sel_t/t0_run1/rows position.j and bounds[j], manifest/audit; reuse exact query and half-open sample coordinates. No new attention/LID/localizer search. |
| E/R_B/old outcome | p2sel.join_gates/P2-R run3; P2-SEL C0/C2 gated D2, NOT C1 historical old direction. Sealed D2/NONE/evaluator metadata. |
| LAC1 pulse | p2sel_e.e1_utterance + p2sel.gated_hook/cached._edit_hook; set g_new from selected sealed LAC0 factor. Steer only original unmasked state/cache. |
| Analysis | p2dir_analyze logit_metrics, draw_weights/boot_stat; p2sel_e/T paired-dialogue patterns. Separate fixed LAC tree; never reuse failed E/T/XA choices. |
| Independent audit | p2sel_t/xa auditor patterns; own candidate/mask/S/F/decision/bootstrap code, no primary-analysis import; p2sel_audit exact bf16 hook emulation for LAC1. |
| Mini decode | cached B/E/S + live D2/readout, canonical metrics and fixed100-ID panel. Conditional additive orchestrator only after passes; no300. |
| Provenance | csasr.utils.provenance/specfreeze + inference_cf manifest utilities; parent/model/tokenizer/processor/code/config/row/input hashes, Git, status/calls/runtime/VRAM. |

## Additive files only in implementation session

Add `src/csasr/inference_cf/lexical_compatibility.py` for pure unmasked candidates, hard-zero mask,
lexical scalar/F math and tiny waveform-input adapter. Add `experiments/inference_cf_p2sel_lac.py`
(candidate seal/manifest/lac0/conditional lac1; later lac2), `_analyze.py`, `_audit.py`, one
`slurm/inference_cf_p2sel_lac.sbatch`, `tests/test_inference_cf_p2sel_lac.py`. Keep helpers narrow;
reuse model/decoder/provenance/evaluator primitives instead of duplicating P2 pipelines.
Do not change readout/core_r2/sites/hooks/legacy reports/configs or method authority.

LAC0: CPU prepare candidate seal from original NONE logits and W*/prefix/model/provider metadata;
commit/push seal before masked outputs. Per utterance load x once, group positions by exact W* bounds,
build masked waveform copy and feature tensor, encode/replay cold B prefix with same cached call shapes.
Use bounded LRU2 masked encoder/branch entries; cache key includes original waveform/model/processor/
mask bounds, branch reuse requires exact prefix lineage. No unmasked/self/cross-key cache transport.
A cold counterfactual needs the WHOLE prefix rebuilt; one evaluation/row can contain many decoder steps.
Never pretend a pre-step original cache with new encoder is natural full-model occlusion.

Record original/local energy in float64, input and outside-slice SHA256 of contiguous float32 bytes,
masked feature/attention-mask hashes, encoder/prefix-cache origin, exact fed tokens and query, candidate
integer IDs/seal hash, raw-logit file hashes and scalars. Enough to independently reproduce masking and
S without serialized masked audio/encoder/KV dumps. If byte-identical mask no-op, reuse original logits,
record no-op and S=0, keep the row. No effect/energy magnitude test determines validity.

LAC2 uses each system's OWN emitted S prefix in unmasked B/E branches. NEW computes ordinary E/R_B/
attention/logits, seals pair and unmasked T-rule W*; when old E*R_B=0 skip counterfactual and edit exactly.
Otherwise mask/re-encode/rebuild the live prefix, keep pair/window fixed, compute S/F, discard masked
trajectory, then original D2/hook on original audio. Reuse current unmasked D2 scratch machinery once,
no LAC backward. Skip D2 when resulting gate zero; log not-evaluated factor versus genuine F=0 separately.
Never reuse teacher-forced per-position pairs/W/F on diverged decoding. No mask reaches final steered
system except through the scalar reference-free F; counterfactual logits do not choose a next token.

## Focused test contract

- Exact ordered180 uid/t/query and old groups42/18/55/5/60; exact T j/W* bounds/heard geometry.
- Pinned unmasked raw seals, accepted all180 XA0 engineering identity evidence, provider/model hashes;
  all80 CPU waveform adapter feature and attention-mask bitwise equality with batch_model_inputs.
- Raw unmasked argmax over exact V_E/V_M; lowest integer ID ties even if list order scrambled;
  candidate IDs sealed/serialization-roundtrip; masked replay cannot reselect from its own logits.
- Mask short/boundary/end windows, shape/dtype/finite preservation, exact positive zeros inside,
  outside-slice byte equality (including negative zeros), energy-positive change and zero/no-op valid.
  No sample padding/cropping/future audio or alternate mask/taper.
- Fresh masked encoder and self/cross caches, full prefix token identity, no target/future token;
  spy tests ensure previous cross-attention states are recomputed, identical-input cache hits only;
  no hot-swap from unmasked cache or stale live prefix. Decoder attention flag/call shapes unchanged.
- Hand-derived M0/Mmask/Delta_E/Delta_M/S identity and invariance to common logit offset; raw logits
  are never rounded probabilities. F exact max0 tanh(S/2), signed/zero/large S and bounds; nonfinite fails.
- torch.inference_mode/frozen params in LAC0; zero backward/D2/LID/steering; runtime feature allowlist
  excludes labels/references/CTC/evaluator margins/gradients/future tokens. No masked generator.
- Synthetic exact LAC0 thresholds (log17/3, log11/9), group support/contrast/CI/draw-count boundaries;
  small S valid NOT_DISCRIMINATIVE, no XA FD rule; one possible factor, no fallback/sweep.
- Existing D2/E/R_B/site/alpha/NormPreserve regression; zero F/gate exact no-edit and bf16 energy;
  LAC1 precedence all branches; independent auditor does not import primary analysis/decision.
- Existing100 panel and live-prefix pair/window/counterfactual lineage, counterfactual call/runtime/
  throughput/VRAM capture; no fresh dataset role/full300.

## Exact implementation / execution sequence

1. Implement additive helper/runner/analysis/independent audit/tests; check→diff→commit→push.
2. CPU source/window/old baseline/provider/partition seals and all80 feature-adapter identity checks;
   prepare candidates_sealed.json, validate all180 IDs/prefixes/W* and push it before masked outcomes.
3. CPU `PASS_TO_P2_SEL_LAC_LAC0` with no masked outcome, immutable manifest on clean pushed sources;
   commit/push pre-run audit. One short sbatch LAC0 counterfactual job, fixed CPU analysis once.
4. Independent `P2_SEL_LAC_AUDIT: PASS (LAC0)` and push rows/logit hashes/statistics/report. INVALID,
   RECALL_ONLY or NOT_DISCRIMINATIVE STOP: no alternate mask/candidate/gate/new diagnostic.
5. Supported only: immutable lac0_selection.json with exact sole F/gate/source/candidate/row/config
   hashes, all predicates and audit; commit/push BEFORE LAC1 outcomes. Pre-LAC1 original-state lineage
   audit, then one C_NEW pulse job reusing C0/C_OLD and sealed F/D2, no new masking/gradient extraction.
6. Independent LAC1 audit PASS and frozen label; every label except GATE_SUPPORTED STOP.
7. Supported only: exact100 panel pre-audit, one conditional B0/OLD/NEW mini decode, independent audit,
   descriptive report. ALWAYS STOP even positive; recommend separately frozen confirmation, never300/P3.
8. At most2 pending/running; limits15/15/120minutes. Preserve partial/failed artifacts; no automatic
   extra jobs. Each reviewed edit test/check→diff→commit→push; final clean/local==origin branch.

No new scientific choices for Claude: pair selection, mask/boundaries, prefix/cache semantics, S/F,
thresholds/uncertainty/precedence, conditional arms, panel, limits and stop rules are all frozen.
