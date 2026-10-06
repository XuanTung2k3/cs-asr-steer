# P2-SEL-E — codebase-aware implementation handoff

This document maps the revised frozen contract in `P2_SEL_E_SPEC.md` to the current repository.
It does not implement an actuator or runner. The initial freeze at `11f4e63` was blocked before
E0 by the preserved audit `P2_SEL_E_PRE_RUN_AUDIT.md`; no E0 outcome existed. This revision fixes
the full-language-softmax Q semantics before outcomes. No E0/E1/E2 job was launched.

## Current-state findings

* `src/csasr/inference_cf/core_r2.py` is canonical for the frozen partition, `Window`,
  `max_attention_window` (50-frame maximum-mass current-query window), `local_support`, and
  `conflict_from_logits`. Keep the original functions unchanged.
* `experiments/inference_cf_p0_r2.py:native_lid` selects `logits[list(language_ids)]` and
  softmaxes that vector. The R2 caller constructs `language_ids` from all unique values of
  `generation_config.lang_to_id`; P0-R2 verifies there are exactly 100 and verifies the EN/ZH
  IDs. P2-R does the same full-map construction; `experiments/inference_cf_cached.py:native_lid`
  delegates to the R2 implementation, and P2-SEL reuses the P2-R E/R_B trace. Hence `P_E/P_M`
  are absolute 100-way probabilities and `Q=P_E+P_M` is the EN/ZH pair mass in `[0,1]`, not 1.
* The frozen model artifact has `generation_config.json` SHA256
  `fbdfa70135de9b1d31553393f14e80aaeb1936ea36576b2ba864055943c09d23`; the sorted token/ID
  mapping digest is `639cd6d6fbdb9cdf9bc09708371676fbc47f98c35f068399d23b3accd9c06731`.
  Already-exposed global P0-R2 Q has n=5,082, q10=0.21689272671937943, q50=0.957465011626482.
  H_E3 tests enrichment of low-Q among the five ZH FP versus 42 EN TP using the fixed global q10,
  predeclared count/contrast rules, dialogue support, and dialogue-bootstrap lower bound.
* `experiments/inference_cf_p2r.py:gate_step` calls the frozen current-query localizer, caches
  native LID by utterance and exact audio crop, applies `local_support`, and returns E/R_B/g and
  four current-window coordinates. The P2-R run3 current traces contain all causal per-token
  E/R_B/g/window scalars. The EN-confusion current traces also contain the pre-existing
  `acoustic.E_oracle` and `timing_error_sec` fields based on P2-R's saved CTC midpoint. These are
  diagnosis-only.
* P2-R traces do not serialize local/null posterior probabilities, attention mass, or the
  maximum-attention frame. One compact extraction job is consequently needed for those quantities
  unless a pre-run artifact audit finds an exact compatible cached record. The E0 job may replay
  the baseline B prefix for the 180 selected queries, use the same audio/encoder inputs, and
  calculate one 0.5 s crop centered inside the frozen 1 s window. It must assert recomputed E
  against P2-R within `1e-6`. It must not run a pulse.
* D2 is sealed in `src/csasr/inference_cf/readout.py` and P2-DIR artifacts; E1 calls the same
  provider semantics. The intervention hook remains DG-02 at L16 with alpha 2 and NormPreserve.
* `experiments/inference_cf_p2sel.py`, P2-SEL JSON rows, report, and auditors establish the exact
  180-key joins, C0/C_old compatibility checks, existing paired dialogue bootstrap, and conditional
  P2-SEL 100-panel. Reuse compatible P2-SEL C0/C_old rows; never infer compatibility from filenames.
* `docs/current/METHOD_CONTRACT.md` is the active authority. This experiment is bounded legacy
  development diagnosis and cannot modify or supersede the core method contract.

## Reuse and minimal additions

| Need | Reuse / implementation instruction |
|---|---|
| E decomposition and original E | `core_r2.local_support`; persist its four input probabilities and exact derived fields. No alternate implementation for E_old. |
| Current localizer | `core_r2.max_attention_window`; capture attention mass and point argmax from the returned `Window`. |
| 0.5 s diagnostic | Add a tiny deterministic crop-bound helper in the P2-SEL-E runner/test only if needed: 8,000 samples centered on the current 1 s window center, clipped to heard duration; never move the center. |
| Previous E | Join `results/inference_cf/p2r/run3/rows/*.json` current steps by utterance and absolute query. Missing history is zero; future query lookup must be rejected. |
| Evaluator-centered diagnosis | Reuse existing EN-confusion `acoustic.E_oracle` and timing fields. No new reference alignment or oracle crop. |
| D2 and hook | Reuse P2-DIR sealed D2 and existing DG-02/NormPreserve execution. Do not alter `readout.py`, direction, alpha, layer, R_B, or hook. |
| Population/outcomes | Reuse P2-RJ positions, P2-DIR extraction/evaluator, P2-SEL old gated rows and bootstrap if hashes/state identities match. |
| Mini decode | Reuse `P2_SEL_MINI_PANEL.json` byte-for-byte and existing cached B/E/S branch/canonical metric infrastructure. |
| Provenance | Use `csasr.utils.provenance` and `csasr.lss.specfreeze`; manifest includes config/spec/design hashes, parent manifest hashes, exact population, model, source tree, run status, and audit links. |

Expected additive implementation files, only after the freeze is pushed and a separate execution
task is authorized: a narrowly scoped `experiments/inference_cf_p2sel_e.py` with E0 extraction and
E1/E2 orchestration; a separate analysis module with the frozen deterministic tree; independent
E0/E1/E2 audit entry points; one Slurm submission file; focused tests. Do not expand or rewrite the
existing P2-SEL runner. No implementation work is part of this design task.

## Contract details the implementation must preserve

E0 has no arms and no steering. It joins the same 180 frozen rows and verifies 42 EN-confusion
positives, 18 EN-confusion misses, 55 ZH-correct negatives, and five ZH-correct false positives.
It recomputes local/null posterior decomposition with epsilon `1e-12`, one short crop, current
attention scalars, causal previous E values, and only already-stored P2-R oracle diagnostics.
Provider-map/hash mismatch, population mismatch, missing rows, or recomputed-current-E difference
above `1e-6` invalidates E0. Q anywhere in `[0,1]` is valid. If and only if H_E3 passes its fixed
global-q10 enrichment criterion, R3 is exactly `E_new=E*Q` and must satisfy `0<=E_new<=E<=1`
within `1e-7`.

The classification order and branch formulas are fully specified in the prose spec/config. In
particular, if E0 supports evaluator-centered or short-only EN-FN recovery, stop with
`P2_SEL_E_LOCALIZER_PRIMARY`: `sqrt(E_long*E_short)` cannot revive a row where the frozen long E is
zero. If static short-vs-long FP suppression and TP retention pass, use exactly that geometric
mean. Temporal repair is exactly `E_t*max(E_(t-1),E_(t-2))`; pair-confidence repair is exactly
`E*Q` with no fitted threshold/coefficient; null repair is exactly a 0.5 shrink of the null
log-odds subtraction. Diagnosis order is H_E1 → H_E2 → H_E3 → H_E4.

E1 has exactly NONE, old-E gated D2, and new-E gated D2. Old rows are reused only if all keys,
state/direction hashes, site, alpha, repair, and evaluator contracts match. The fixed six-endpoint
90% simultaneous dialogue-bootstrap family and label precedence cannot be changed after E0. If
E0 is ambiguous, localizer-primary, or invalid, no E1 job is permitted.

E2 is permitted only after `P2_SEL_E_REPAIR_SUPPORTED` and independent E1 audit PASS. The exact
P2-SEL 100 IDs and SHA are reused. It is descriptive development evidence and always ends P2-SEL-E;
no full 300, P3, or fresh validation role is authorized.

## Pre-run and post-run audit

Before E0, CPU audit must verify P2-SEL audit/report terminal status, exact position and source
hashes, P2-R joins, D2 seal, frozen 100-language mapping/hash, EN/ZH token IDs, existing P2-SEL
panel hash, E0 zero-outcome state, exact constants, and role firewall; it must not constrain Q to
be near 1. Before E1, independently verify E0 audit PASS, exactly
one selected branch, formula hash, all E0 rows, and C0/C_old reuse lineage. Before E2, verify E1
support plus audit PASS and exact panel identity.

The independent post auditor must not import the analysis/decision module. It independently
recomputes E decomposition, expected groups and dialogue IDs, causal joins, hypothesis booleans,
precedence, exact E_new values, actual hook energy, margins/corruptions, bootstrap bounds, E1
label, and if allowed the E2 panel/metric lineage. Required terminal audit string is
`P2_SEL_E_AUDIT: PASS`; any failure forbids a scientific conclusion and stops later stages.

## Data and compute firewall

Allowed inputs are only already-exposed D-dev-select and P2-R/P2-RJ/P2-DIR/P2-SEL artifacts.
Disallow new router-calib role, D-dev-confirm, D-test, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND,
Qwen transfer, or new validation. Evaluator/CTC alignment is diagnostic-only. E0 uses CPU unless
the missing local/null probabilities and attention scalars require the one permitted small MIG
job. E1 is at most one MIG job. E2 is at most one conditional MIG job. `sbatch` only, at most two
pending/running jobs, no job from the design freeze.
