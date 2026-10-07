# P2-TTA0 — implementation-ready handoff

Design/config/panel frozen pre-outcome. Scientific decisions in SPEC and JSON, no alternatives.
Do not reopen terminal steering diagnostics or relabel TTA as the active v6 core method.

## Actual reuse map

- csasr.models.whisper: local bf16/eager/eval snapshot, preprocessing and encoder; encoder frozen.
- cached.Branch.step / core_p1.processed_argmax: ordinary forced decoder, isolated fresh KV,
  attentionTrue/passive L16 recorder. Never cached_decode/seq_decode adaptation with gate/D2.
- P2-SEQ S0 rows and sealed AUTO reuse proof:20 exact pseudo token lists; model/input/library hashes.
- P2-SEQ analysis outside_lexical/utt_counts and canonical metrics/retention: evaluation only.
- p2seq_audit templates: independent count/bootstrap/manifest/reset/status auditing patterns.
- Installed Whisper model/config:32 layers/1280 width, three LN modules per layer plus final decoder
  LN; complete194 parameter names in p2_tta0.json,248320 scalars (0.0160882%). Meta-only inspection
  and synthetic LayerNorm gradient test were done without model outcomes in design session.

No existing episodic-TTA runner/objective exists. Add src/csasr/inference_cf/episodic_tta.py,
experiments/inference_cf_p2tta0.py, inference_cf_p2tta0_analyze.py, inference_cf_p2tta0_audit.py,
slurm/inference_cf_p2tta0.sbatch, tests/test_inference_cf_p2tta0.py. Preserve loader/provider modules,
historical reports/results and core METHOD_CONTRACT. Output results/inference_cf/p2tta0/run1,
P2_TTA0_REPORT.md, and selected-only evidence handoff P2_TTA0_TTA1_HANDOFF.md.

## Exact sequence for Claude

1. Implement enumerated-LN fp32 masters and reset guard. Resident model stays bf16/gradient-free;
   functional_call overrides selected LN values with differentiable bf16 casts, never detached.
   AdamW optimizes fp32 masters only. Master checkpoint/optimizer fresh per objective. Final decode
   copies effective bf16 values into selected names only, then finally bitwise restores theta0.
2. Implement manual unmasked teacher queries: input cB+y, raw logits3+t, fixed valid positions,
   exact decoding suppression; float32 allowed log_softmax; entropy for A1, target NLL for A2.
   EOS not a loss target but remains predictive class. Built-in labels loss is not this contract.
3. CPU seal first-in-existing-order20 IDs/pseudo lists and reuse proof from audited rows. No
   reference dependency in inference modules. Same teacher kept across both steps, no sampling.
4. Add common-y_B diagnostics/loss/update/reset logs and canonical evaluator-side metrics. Severe
   ending detector depends only baseline/adapted content length and EOS, not reference timing.
   Independent auditor owns criteria/selection/bootstrap and first-utterance loss/grad check.
5. Focused tests below, CPU preaudit PASS_TO_P2_TTA0. Resolve model/files/env/git/config/audio/
   teacher/reset/suppression provenance, push all code/seals/manifest/audit before scientific job.
6. Submit ONE sbatch2h maximum (target<1h). Each utterance one detached ordinary encoder clone;
   verify ordinary theta0 forced decoder equals cached S0 before adapting. A1: reset/create masters,
   L0/update1,L1/update2,L2, materialize/decode,finally reset. A2 from fresh theta0: same schedule,
   own teacher and common-y_B final diagnostics,decode/reset. No encoder graph/cache persistence.
7. CPU canonical analysis after outputs sealed, independent P2_TTA0_AUDIT: PASS. Commit/push report,
   raw compact evidence/checkpoints/hashes/status. Apply frozen independent labels and one objective
   selection only. STOP, no100/300/P3. Failure is not permission for another objective/LR.

## Required focused tests

- Panel byte hash/exact first-per-dialogue IDs,20 unique dialogues, parent unchanged.
- Actual named_modules/LayerNorm affines exactly match all194 names/shapes/count/fraction.
- Only selected fp32 masters trainable; resident encoder/non-LN no grad/update; functional-call
  original parameter objects/flags restored and no model gradient contamination.
- Synthetic tiny Whisper theta0 functional logits bitwise equal ordinary bf16 forward; gradient
  reaches masters through bf16 casts; zero master update ordinary decode same. No fp32-forward drift.
- Reset original LN bf16 and new masters fp32 bitwise, fresh optimizer moments/cache, finally on error.
- A1 entropy vs hand-computed distribution including EOS; A2 NLL vs teacher token; suppression0/
  later indexed correctly, prefix/finalEOS query excluded, no labels shift/off-by-one.
- Empty teacher2 no-ops; cap200 content preserved; disallowed/interior-special cases explicit; fixed
  valid-position mask, no future/reference/evaluator teacher, AUTO detached/unmodified.
- Optimizer class/settings/exact2 steps, no clipping/scheduler/accumulation; loss evaluated3 times.
- No steering/D2/LID/gate calls, frozen greedy bf16 forced decode and AUTO teacher semantics.
- Severe truncation boundary, lexical outside harm vs harmless transcript edits, canonical POI
  transition identity, deterministic viability/confirmation-bias/selection including exact ties.

Use existing test_canonical_metrics/test_inference_cf_p2seq and model/decoder tests as primitives.
No scientific GPU acceptance job before main allocation: same-allocation theta0 identity and
independent first-row objective/gradient checks invalidate rather than trigger alternatives.

## Audit/manifest and numerical boundaries

Use provenance/specfreeze conventions; resolved config/model/tokenizer/env/source/input hashes,
git clean tree/commit, original theta0 hashes, pseudo seal/reuse proof, job/system timing/status,
all rows and optional tiny checkpoints hashed. Full audio hash and existing fingerprint have
DIFFERENT explicit labels; do not repeat prior64KiB/raw-sha mismatch.

Log teacher per-position entropy/selected logprob, mask counts, common-y_B entropy/script/EOS
summaries, gradients/losses/updates and actual effective changed weights. Independent CPU analysis
recomputes aggregate loss and all metrics. First-row live auditor independently verifies each
objective's actual loss/gradient once at theta0, stores scalar comparison; no extra optimizer
updates. Immutable theta0 and LN/master checkpoint hashes verify resets/updates; full non-LN
baseline hash at load/final and parameter-version evidence per episode, no giant model copies.

BF16-cast masters accumulate fp32 updates even if intermediate casts round them away; report
both master and effective norm, do not change precision/steps to get effects. A missing/nonfinite
loss/grad/update/logit, reset failure, baseline mismatch or incomplete episode is stage INVALID.
No lower learning rate, clipping, new teacher or parameter subset fallback.

At most80 optimizer updates,82 backwards including independent first-row checks,142 teacher
forwards (120 loss+20 common-A2+2 audit),40 final decodes and20 original forced identity decodes.
Baseline/AUTO timing marked reused, not estimated from other runs. Fresh KV each weight state.
Empty/no-valid episode still has2 no-op updates and identity output, no drop/refill.
Selection only from viability-passing objectives, then lower MER/PIER/ZH-CER/master delta norm;
exact tie A2. All screen evidence is development only. Later100 TTA1 needs its own frozen contract.
