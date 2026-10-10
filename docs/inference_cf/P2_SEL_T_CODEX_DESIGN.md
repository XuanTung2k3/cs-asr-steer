# P2-SEL-T — implementation handoff

Contract: P2_SEL_T_SPEC.md and configs/inference_cf/p2_sel_t.json. This design session changes documents
and configuration only. Parent 2810cf2 is clean/ahead-of-none and preserves the audited P2-SEL-E ambiguous
stop. This human-authorized stage does not revise METHOD_CONTRACT or historical results.

## Reuse map and bounded implementation

| Need | Existing code/artifacts and exact instruction |
|---|---|
| Population/group | p2rj/positions.json; p2dir/construction_population.json; p2sel.join_gates; old E0 group_of. Unique uid/t/query joins, historical counts42/18/55/5/60. |
| Baseline replay/current attention | p2r.DiagBranch (cached.Branch); last-query cross-attention over frozen generation_config.alignment_heads. Retain existing model/encoder/prefix/precision. |
| W_1s | core_r2.max_attention_window unchanged. Compare live W/query/mass with audited E0 rows. |
| E_1s/CENTER/null | p2sel_e/e0_run1 JSON rows and manifest/null; p2sel_e.short_bounds, decompose, core_r2.local_support. Exact CENTER bounds/provider/null identity checked before reuse. No new long/center/null calls. |
| New work | two fixed LEFT/RIGHT native_lid calls/key; compute all3 masses from normalized heard-frame means; deterministic j*. Store small mean-attention vectors for audit. |
| Old causal outcomes | p2sel/s1_run1 *_a rows C2 gated D2, not C1 OLD direction; p2dir NONE/D2 sealed vectors/evaluator. Reuse exact lineage; no old extraction. |
| T1 new pulse | p2sel_e.e1_utterance pattern, p2sel.gated_hook, unchanged cached._edit_hook/site/NormPreserve. New gate only; existing sealed D2. |
| Evaluation/bootstrap | p2dir_analyze.logit_metrics/draw_weights/boot_stat, p2sel_e paired group summaries; implement T criteria separately, not inherited E tree. |
| Auditor | extend p2sel_e_audit pattern minimally; separate T decision/stat calculations; never import primary T decision module. Exact bf16 energy emulation from p2sel_audit. |
| T2 | cached B/E/S primitives + readout live scratch computation, canonical metrics/provenance. No existing P2-SEL S2 runner exists; a conditional additive orchestrator is needed only after passes. Preserve live prefix/cache isolation. |
| Panel | docs/inference_cf/P2_SEL_MINI_PANEL.json byte-identical; no new IDs. |

Add narrowly scoped `experiments/inference_cf_p2sel_t.py`, `_analyze.py`, `_audit.py`, one
`slurm/inference_cf_p2sel_t.sbatch`, and `tests/test_inference_cf_p2sel_t.py` in an implementation session.
Reuse helpers; do not duplicate P2/R pipelines or create another audit framework. Do not alter historical
core_r2/readout/sites/hooks or P2-SEL/E reports/configs. Any genuine incompatibility blocks execution.
The original E-stage R3/E1 conflict is irrelevant: no E repair branches are inherited here.

T0 runtime input allowlist: model/encoder, current B cache/query/prefix, available waveform length/identity,
head map, W/sample geometry, language-ID map, null probabilities, eps. Output E_tok must accept no group,
reference/CTC/evaluator timing/margin, future token, or future audio. Group labels enter CPU analysis only.
No C_tokctx steering input or fallback. Exact ties choose L then C then R. Missing/invalid candidates
invalidate the stage, never select another signal. Save compact per-key bounds/pi_E/pi_M/E/mass/selected
fields and NPZ mean vectors; hash raw probabilities, null, model/head map and parent artifacts.

## Focused test contract

- Exact ordered180 keys/uid-t-query joins and historical42/18/55/5/60 groups; no missing/duplicate rows.
- Frozen100 language mapping/source/null identity, E_1s within1e-6 of P2-R/P2-SEL; CENTER raw probability
  reuse and exact bounds equal E0 short_bounds for every exposed key.
- Bounds at exact1s, shorter-than0.5s, partial last-frame, clipped utterance end; all within W/heard;
  no off-by-one; float64 overlap masses against hand-derived vectors; head average/normalization/query
  source unchanged; all three full identical short-W masses equal and earliest L wins.
- Exact maximum ties L/C/R; chosen E equals selected candidate bitwise; selection independent of E values;
  E_ctx excludes selected index; overlapping descriptive normalization never treated as disjoint mass.
- Feature allowlist rejects reference/evaluator/future information; C_tokctx never enters repair; zero gate
  exact NONE; frozen D2/R_B/site/alpha regression, cache/param isolation reuse existing tests.
- Synthetic predicates cover all T0/T1 boundaries/precedence, dialogue support, missing bootstrap draws;
  supported only one R_TOK; contrast/recall labels STOP; no alternate-repair fallback.
- Exact existing mini-panel hash/IDs and live-prefix (not stored diagnostic E) T2 feature behavior.
- Auditor independent from decision module and independently reproduces crop/mass/energy/stat verdicts.

## Exact execution sequence and stops

1. Implement only additive helpers/orchestration, focused tests, independent audit; check/diff/commit/push.
2. Build immutable manifest from pushed clean sources; CPU pre-T0 audit PASS_TO_P2_SEL_T_T0 and push it.
3. One short sbatch T0, no steering/readout; analyze once and independent audit PASS (T0); push results.
4. Any T0 label other than TOKEN_LOCALIZATION_SUPPORTED ends stage. No contrast formula invention.
5. If supported, create t0_selection.json with sole formula E_new=E_tok, row/attention/source/config hashes,
   full predicates and audit links; commit/push BEFORE T1. CPU pre-T1 lineage audit, then one C_NEW job.
6. Independent T1 PASS and label. Any label other than P2_SEL_T_TOKEN_GATE_SUPPORTED ends stage.
7. Only supported permits exact100-panel B0/OLD/NEW mini decode, one job, no BROAD/AUTO regeneration.
   Independent T2 audit and report; always STOP. No300/P3/newrole/transfer.
8. Every reviewed change checks → diff → commit → push; final clean/local==origin branch.

Engineering INVALID and time-budget exhaustion stop for review, not another scientific job. All numeric
criteria, precedence, bootstrap and budgets are in the spec/config; Claude must not choose new scientific
thresholds, candidate windows, repair formulas, populations or outcome-driven exceptions.
