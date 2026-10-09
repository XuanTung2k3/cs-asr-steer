# SRC-CF0 — source review and blocked implementation plan

Status: **SRC_CF0_DESIGN_BLOCKED**, not a runnable scientific freeze. See SRC_CF0_SPEC.md for
the verified correct-Mandarin coverage limitation. No model forward, new vector, pulse,
lexical evaluation, runner, Slurm script or production command was created.

## Reusable interfaces actually inspected

| Component | Existing interface | Constraint |
|---|---|---|
| Original / masked cached replay | `experiments.inference_cf_cached.Branch.step`, `.crop`, `_processed` | Cold owner cache; exact S1 M path and suppression; no masked/original KV sharing |
| Original encoder / masks | `experiments.acoustic_s1.encode`, `candidate_utterance`, `acoustic_utterance` | Reuse primitives only; candidate-union reranking is not the new direction experiment |
| Predicted association | `csasr.inference_cf.s1_evidence.heard_attention`, `select_target`, `select_offtarget` | Exact rational integral/tie guards; no relaxed coverage or candidate-dependent association |
| Primary predicted track | `csasr.inference_cf.r0_regions.query_mapping`, `intervals_to_track` | R0 primary heard intervals only; no oracle/script/SVD arrays |
| Waveform intervention | `csasr.inference_cf.lexical_compatibility.hard_mask`, `waveform_model_inputs` | In-memory positive zero, full feature/encoder recompute |
| Native DG-02 capture | `csasr.lss.sites.DecoderPostCrossAttnRecorder` | Native residual q+u_source; validate exact tensor consumed by final_layer_norm |
| DG-02 pulse | `DecoderPostCrossAttnInterventionHook`, `loc0_sites.cross_pulse_hook` | Pre-FFN only; no legacy post-block hook or depth scaling |
| Relative pulse dose | `csasr.inference_cf.prompt_r1.relative_action` | Audit reuse against authorized amendment A1; do not inherit the extra norm-rounding rejection |
| Chord solver | `experiments.inference_cf_p2r.solve_scale`, `scaled_direction`, `emulate_edit_norm` | Native dtype, bounded solve, sign-specific reachability |
| Repair | `csasr.models.hooks.apply_steering` | NormPreserve, EPS=1e-6, zero-dose bitwise no-op |
| Historical apparatus | `experiments.inference_cf_st_loc0.Engine` and sealed ST-LOC0 cells | Reproduce clean baseline/site/control, not its obsolete direction provider |

S1 stores logits and attention, not the paired DG-02 hidden states required for v_AC. Its masked
logits cannot substitute for hidden representations. R0 uses PATH5 baseline prefixes and full
replay; those states are not interchangeable with the original B0M_L16 cached path. S1 numeric
scores, candidate membership and gold-accessible populations cannot restrict the direction or
the full-vocabulary steering readout. Reusing a dimension-compatible tensor is insufficient.

The installed Whisper residual geometry is q=self residual, r=q+cross-attention output, then
r+FFN(LN_final(r)). Record/steer r; never the whole decoder block output. Native residual rounding
and FFN-input consumption require explicit audits, not a float32 approximation. Capturing both
layers in the same unedited replay should be possible; validate unchanged logits first.

## Prospective isolated architecture — not implemented or authorized

After a separately approved resolution and complete freeze only:
- `cf_direction.py`: pure paired-state contrast/normalization/geometry; no evaluator import.
- `inference_cf_src_cf0.py`: preparation, reference-free construction, seal, conditional isolated
  pulses, output seal; reuse S1 waveform/cache and DG-02 interfaces, not its reranking driver.
- Separate evaluator and independent auditor. Auditor may share generic IO but not primary
  vector, selection or terminal logic. No scientific audit PASS is claimed by current CPU checks.
- Future model adapter: identify model/site, encode each audio, replay identical prefix, return
  native residual plus input/cache lineage. Whisper is the only permitted implementation;
  no speech-LLM experiment or adapter is built now.

Required future barriers would include all180 clean logits bitwise historical NONE; identical
UID/audio/prefix/query; native post-cross residual reconstruction and FFN consumption; exact
zero-dose; cache/hook restoration; unchanged weights; historical apparatus replay; and source,
tokenizer, preprocessing, region/waveform/model hashes. D2 may use its sealed reference-free
readout vectors for apparatus verification; no new backward call is permissible.

Controls would require NONE/zero, eight signed target arms and matched signed off-target arms,
deterministic matched-energy random arms, and compatible historical prompt results descriptive
only. Target-only and paired comparisons must remain separate. No oracle union can advance.
Their final seed/family/statistical gates are not silently filled in while the design is blocked.

## Resource forecast, not authorization

Future construction could group the same S1 80 original utterances and 124 unique masks (132
masked query readouts); capture L16/L24 simultaneously, then repeat for reproducibility. The S1
original M path used3159 cached steps and masks3500. Two repeats would be about13318 cached
steps and408 encoder calls, excluding additional integrity barriers. B would have at most
1440 primary cells plus off-target/random/zero/apparatus cells, each isolated at one query.
Historical S1 phases took1–2 minutes and ST-PROMPT-R1 with6480 primary/random cells took6.3 minutes.
A conservative forecast would be5–15 minutes per phase, <=2 sequential jobs, <=3h/job after
an approved freeze and throughput preflight. No job or compute revision is authorized now.

## Review artifacts and tests

`SRC_CF0_PANEL.json` fingerprints the unchanged population, all80 original S1 region JSON records,
S1 seals/audits and reusable sources. `configs/inference_cf/src_cf0.json` is a **blocked review
record**, with execution_authorized=false and scientific_gates_frozen=false. It is not a runner
configuration. CPU tests independently recompute coverage and verify seal files/panel hashes.
No new SRC-CF0 outcome report is created, and all historical files remain unchanged.

CPU validation: 10 blocked-design checks, 91 S1 freeze checks, 21 S1 implementation checks,
and 51 focused P2-R / ST-PROMPT-R1 / ST-LOC0 / DG-02 regressions passed (173 unique tests).
The broader regression invocation was interrupted; the final focused invocation deselected
the expensive historical P2-R population rebuild and passed. No pretrained-model inference
was used. Synthetic-model engineering tests are distinct from scientific outcomes. Joblib
reported unavailable multiprocessing shared memory and ran serially; there were no test failures
in the completed suites. `git diff --check` passed, and STATUS retains its entire original byte prefix.
