# P2-SEL — implementation handoff

This is the codebase-aware implementation map for the frozen contract in
`P2_SEL_SPEC.md`. No runner or actuator is implemented here. P2-SEL has no scientific outcome
yet; do not start a job until the implementation and independent pre-run audit satisfy this
document and the spec.

## Starting evidence and reconciled local drafts

At design start, local branch `feature/inference-cf-steering`, local and remote
`cs-asr-steer-inf` both pointed to `6e480b777086cb963ff3cf77b730ca25167e6bc2`; working tree had
only untracked `configs/inference_cf/p2_sel.json` and `docs/inference_cf/P2_SEL_MINI_PANEL.json`;
Slurm queue was empty. The config's old `design_base_commit` (`c899fa1…`) was stale, and it
referenced the absent `P2_SEL_SPEC.md`. Its stage status was premature. These are corrected by
this freeze. Its D2/site/gate/alpha/population/arm definitions and source anchors were retained
where they matched inspected code.

The mini-panel draft's 100 selected stable IDs and per-ID selection hashes were retained. The
draft's recorded role-mapping file hash (`ec7a4f…`) differed from the current external Parquet
file hash (`25a45f…`). A read-only semantic audit with the current `acl1` PyArrow environment
verified all 300 parent-panel IDs against the current role file: every ID is still D-dev-select,
all 300 dialogue IDs match the recorded eligible-ID map, the eligible-ID set equals the parent
P2-A panel ID set, and every selected ID is in that set. The panel JSON now records the current
role-mapping file hash; its canonical payload and file hashes were recomputed. No row selection
changed. The final file SHA256 and canonical panel digest are recorded in the config. No labels
or outcomes were consulted to select utterances.

No scientific P2-SEL job, P3 run, fresh validation role, new direction, or search over layer,
alpha, gate, or localizer has occurred. The frozen panel is a subset of already exposed
D-dev-select and therefore remains development-only.

## Reuse map

| Purpose | Reuse |
|---|---|
| D2 construction | `src/csasr/inference_cf/readout.py:readout_direction`, version `p2dir_readout_v1`; use sealed D2 vectors in S1 and runtime provider in S2. Keep its inputs reference-free, scratch cache isolated, params frozen, and autograd limited to the zero leaf. |
| Gate/localizer | `src/csasr/inference_cf/core_r2.py:tokenizer_partition`, `conflict_from_logits`, `max_attention_window`, `local_support`, `gate_values`; raw `E`, `R_B`, and current gate rows in `results/inference_cf/p2r/run3`. Recompute `g=E*R_B` and audit against the stored value. |
| OLD direction | `src/csasr/inference_cf/core_p1.py:direction`; call unchanged on matched B/E site vectors. |
| Site/repair | `src/csasr/lss/sites.py:DecoderPostCrossAttnInterventionHook` and `src/csasr/models/hooks.py:apply_steering`; same DG-02 site and NormPreserve semantics. |
| Pulse mechanics | `experiments/inference_cf_p2r.py:solve_scale`, `pulse_hook`, and `DiagBranch`; use exact hook dtype/chord solver behavior and eight-evaluation cap. |
| Fixed population and D2 ungated reference | `results/inference_cf/p2rj/positions.json` and audited `results/inference_cf/p2dir/exp1_run1/` rows, sealed directions and logits. Reuse only after hashes, key order, state identity, and P2-DIR audit pass. |
| Cached free decode | `experiments/inference_cf_cached.py:Branch`, `_edit_hook`, and `cached_decode` as the generation/branch/cache foundation. Add D2 orchestration around the existing B/E/S stepping without changing OLD behavior or persistent cache semantics. |
| S2 BROAD | `src/csasr/inference_cf/broad.py:BroadBudget` and `energy_match`; apply unchanged per-utterance packet/debit rule. |
| Metrics/retention | `src/csasr/evaluation/canonical.py`, `retention.py`, `result_emit.py`; use canonical PIER/MER/EN-WER/ZH-CER, POI transitions, candidate harm, and language retention. |
| Bootstrap and independent audit patterns | `experiments/inference_cf_p2dir_analyze.py` and `experiments/inference_cf_p2dir_audit.py`; reuse data formats and bootstrap draw convention, but write P2-SEL stage-specific analysis/audit with the frozen P2-SEL thresholds. The independent auditor must not import the primary analysis/decision module. |
| Manifest/Slurm | `csasr.utils.provenance`, `csasr.lss.specfreeze`, and `slurm/inference_cf_p2dir.sbatch`; follow repository manifest and MIG Slurm patterns, with a P2-SEL pre-run gate. |

P2-R run3 covers 80 utterances and provides gate signals for its current queries; join its row
steps to the exact 180 position keys using utterance ID plus absolute query. Missing or duplicate
join keys invalidate S1. P2-DIR extraction verified its 180 state/query population against P2-R.
Do not treat P2-A's old gated free-decode output as C1: S1 C1 is an independent single-pulse
arm and must be executed unless an auditor proves exact pulse-level semantic identity, which no
current artifact establishes.

## Files to add for implementation

- `experiments/inference_cf_p2sel.py`: prepare/extract if necessary, S1 C1/C2/C3 single-pulse
  executor and conditional S2 cached greedy orchestration. Keep reference/evaluator functions
  after direction and gate artifacts are sealed.
- `experiments/inference_cf_p2sel_analyze.py`: compact row analysis, exact paired dialogue
  bootstrap, fixed S1/S2 label precedence and descriptive scalar distributions.
- `experiments/inference_cf_p2sel_audit.py`: CPU pre-run and independent post-run audit; must not
  import the analysis or decision implementation.
- `slurm/inference_cf_p2sel.sbatch`: one S1 job and conditionally one S2 job, H100 MIG, manifest
  first, audit gate required.
- Focused P2-SEL tests only for new integration: exact run3 gate joins, D2 call timing/cache
  isolation, matched-state pulses, pooled S1 BROAD energy, S1 thresholds/precedence, deterministic
  panel verification, S2 per-utterance BROAD accounting, and auditor independence.

Modify only P2-SEL runner/analysis/audit/Slurm additions and any minimal shared cached-decode
integration required for D2. Leave `readout.py`, `core_r2.py`, `core_p1.py`, DG-02 hook/repair,
P2-DIR artifacts/reports/spec/config/audits, canonical metrics, historical P2 output, and
`docs/current/METHOD_CONTRACT.md` unchanged. If implementation requires changing any frozen
component or these protected files, stop and report an implementation gap before running.

## Exact execution sequence and stop rules

1. Verify the committed spec/config/panel hashes; verify current git state and zero P2-SEL
   outcomes. Run only CPU pre-run audit. It must emit `PASS_TO_P2_SEL_S1`; otherwise stop.
2. Execute one S1 Slurm job. Reuse P2-DIR NONE/ungated-D2 outcomes and sealed D2 vectors after
   compatibility checks. Execute C1/C2/C3 single pulses only. Save compact row scalars/states
   sufficient to reconstruct energy and margins; no new high-dimensional dumps.
3. Run the independent S1 audit. If it is not `P2_SEL_AUDIT: PASS (S1)`, label invalid and
   stop. Otherwise compute the frozen label exactly once from the frozen thresholds.
4. For every S1 label except `P2_SEL_GATE_RESCUES_D2`, stop. When stopped, report only the
   prescribed by-stratum E/R_B/g/dose summaries; no GPU follow-up or localization diagnosis.
5. Only after S1 rescue and audit PASS, submit one S2 Slurm job over the exact frozen 100 IDs.
   Reuse exact compatible B0/OLD/AUTO caches if proven by manifests; otherwise regenerate only
   required systems in that job. AUTO is omitted if no exact cache exists. Reuse NEW actual
   realized energy as the reference-free per-utterance BROAD budget.
6. Run the independent S2 audit. Apply the frozen mini-screen label and stop, including after a
   promising result. Recommend a separately frozen full 300-utterance development confirmation;
   do not launch it here.

Across P2-SEL, maximum is two pending/running GPU jobs, one S1 and at most one conditional S2;
target total is at most two GPU-hours. No router-calib, D-dev-confirm, D-test, P3, or transfer
data. Every executed run requires the existing provenance manifest helpers and explicit
`P2_SEL_AUDIT: PASS` before a scientific interpretation is reported.
