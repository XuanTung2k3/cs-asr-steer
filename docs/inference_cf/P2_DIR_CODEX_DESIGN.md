# P2-DIR codebase-aware implementation handoff

Design base: `ae6d10b9f7194a7e090b52f95dc49ab63a407a18`, clean and equal to fetched
origin/cs-asr-steer-inf, 2026-10-06. No running/pending Slurm jobs at start.
Specification: `P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md`; configuration:
`configs/inference_cf/p2_dir_direction_identification.json`. These are the scientific
freeze. No new outcome has been computed. P3 remains held. Do not make additional scientific
choices. If an invariant cannot be implemented, stop with a named implementation gap.

## Actual entry-point trace and reuse

| Capability | Actual path and callable | Responsibility |
|---|---|---|
| R2 provider | `src/csasr/inference_cf/core_r2.py` | tokenizer_partition, max_attention_window, local_support, conflict_from_logits, gate_values; all remain unchanged |
| Original reference-free gate runner | `experiments/inference_cf_p0_r2.py` | native local LID/provider context; reference-free panel |
| P1 direction | `src/csasr/inference_cf/core_p1.py:direction` | frozen OLD epsilon/precision/fallback, selected_gate, processed_argmax |
| Exact actuator | `src/csasr/lss/sites.py:DecoderPostCrossAttnInterventionHook`, `src/csasr/models/hooks.py:apply_steering` | existing exact-site repair; unchanged |
| Deployment P2 | `experiments/inference_cf_p2.py:run_utterance` -> `experiments/inference_cf_cached.py:cached_decode` -> Branch.step/_edit_hook | B/E/S cache semantics, shared generated prefix, S edited lineage |
| P2 controls/manifests | `experiments/inference_cf_p2_prepare.py`, `inference_cf_ce.py`, `inference_cf_ce_accept.py` | clean Git/frozen panels, cached equivalence |
| P2-R pulses | `experiments/inference_cf_p2r.py:DiagBranch,pulse_hook,pulse_pass,solve_scale,scaled_direction,summarize` | matched-state, exact realized bf16 energy; reuse diagnostic orchestration without oracle relocation |
| P2-R replay | same module `gate_step,replay` | frozen gate/dose persistent S replay for Exp2 |
| Population | `experiments/inference_cf_p2r_population.py`, `inference_cf_p2rj_prepare.py:build_positions` | frozen parent population and reference-token/competitor metadata; evaluator only |
| Useful evaluator Jacobian | `experiments/inference_cf_p2rj.py:SiteGradientProbe,grad_step,evaluator_targets,position_stats` | evaluator only; NEVER imported into production direction provider |
| Final population expansion | `experiments/inference_cf_p2rje.py:eligible_en_confusion,build_positions,competitor_prepass` | ended diagnostic program; read historical report only; do not expand P2-DIR |
| Unique math | `src/csasr/experiments/basis_a5_unique_shared.py` | reusable numerical concepts/test fixtures; production top_vectors behavior is NOT reused |
| Historical extraction | `experiments/basis_a5_unique_shared.py:construct_whisper,_save_directions,_preceding_indices` | proves actual populations/timing/rank, remains untouched |
| Evaluation/bootstrap | `experiments/inference_cf_p2_evaluate.py:metrics,flips,per_utt_counts,paired_bootstrap`; `src/csasr/evaluation/{canonical,retention}.py` | canonical transcript counts; add new fixed-family bootstrap, not old P2 selection rules |
| Existing independent audit examples | `experiments/inference_cf_p2{,r,rj,rje}_audit.py` | provenance/row completeness patterns; independent P2-DIR decisions required |
| Slurm | `slurm/inference_cf_{p2,p2r,p2rj,p2rje}.sbatch` | MIG environment, acl1 interpreter, offline models; new conditional launcher |
| Provenance | `src/csasr/utils/provenance.py`, `src/csasr/lss/specfreeze.py`, inference_cf core hashing | resolved configs/model/environment/git/data/source/artifact hashes |

## Common interface and cache lifecycle

Add `src/csasr/inference_cf/directions.py` (suggested exact file) with independent providers
OLD/UNIQUE/READOUT. Provider input is a reference-free context containing current unedited B site,
E site if required, pre-step B scratch cache, encoder outputs, current tokens/cache_position,
suppression/script partition and sealed static vector. No evaluator targets or labels.
Return `{direction, status, reason, provenance, counters}`; direction detached float32 D-vector,
status explicit, no implicit fallthrough to another provider. Constructor fold selection stays
in manifest/panel infrastructure, not in runtime linguistic features. Direction providers do not
own dose/energy decisions. Caller chooses frozen deployment dose or diagnostic chord solver.
OLD adapter delegates core_p1.direction exactly. Unit checks allow OLD epsilon deviation.

Add `src/csasr/inference_cf/unique.py`: explicit top32-only numerical construction/sealed fold
loader. Build moments only from approved baseline representations. Do not change A5 builder to
fix its historical mismatch. Provider reads vector only, no reference artifacts.
Add `src/csasr/inference_cf/readout.py`: private reference-free zero-probe scratch forward, no
P2-RJ imports. Capture pre-step cache BEFORE B advances; clone inference tensors into ordinary
non-inference detached tensors. Own context managers for hooks and cleanup. Scratch autograd
cannot advance persistent caches. D2 construction must be value-identical to B; S consumes its
result but retains its own historical edits. No differentiation through earlier tokens.

Modify `experiments/inference_cf_cached.py` only to accept an optional provider and acquire a
pre-step scratch context when needed. Default provider=None must execute historical OLD path
without changed flags/forwards/cache semantics. Existing P2 runner and scientific configs remain
unchanged. NEW decode may omit unused E forwards only if full OLD regression remains exact;
freeze actual counters/manifests, no gate changes. D2 alpha0/g0 skips backward.

Add `experiments/inference_cf_p2dir_prepare.py`: source/config hashes, immutable approved IDs,
CPU fold construction from extraction states, manifests. Add `inference_cf_p2dir.py`: separate
extract/exp1/exp2/exp3 modes, guarded by predecessor audit hashes; reuse P2-R and cached kernels.
Add `inference_cf_p2dir_analyze.py`: pure fixed statistics/decisions; evaluator-side reference
loading only. Add `inference_cf_p2dir_audit.py`: independent recomputation, no analyze imports.
Add `slurm/inference_cf_p2dir.sbatch` with explicit STAGE/OUT and predecessor PASS guards.
Add `tests/test_inference_cf_p2dir_{directions,protocol,audit}.py` covering spec section10.
Outputs exclusively `results/inference_cf/p2dir/` with immutable per-attempt directories.

Forbidden modifications: existing results/out/legacy files; historical diagnostic configs/specs/
reports/populations; P2 frozen selection; core_p1/core_r2 math; sites/apply_steering; DG-03R/DG-08
scientific artifacts; model/tokenizer/suppression; split assignments; blocked role content.
Any necessity to alter frozen primitives is a stop-and-report implementation gap, not a routine fix.
New documents/current rollup additions are permitted; do not overwrite historical paragraphs.

## Exact implementation and execution sequence

1. Verify HEAD/status/remote and frozen spec/config hashes. Preserve newer local work. Read spec.
2. Implement pure providers and synthetic focused tests. Keep OLD unchanged. Implement manifest,
   firewall, independent analyzer/auditor and all threshold fixtures before GPU outcomes.
3. Add common-provider integration; run existing cached/CE/P1/P2/P2-R and DG-02 CPU tests.
   Validate D2 inference-mode boundary and exception cleanup. Review diffs, commit, push each
   reviewed change batch to origin HEAD:cs-asr-steer-inf. No force push/merge main/PR.
4. CPU prepare frozen180 IDs and approved extraction plan. Pre-run independent audit
   PASS_TO_P2_DIR_RUN; seal implementation HEAD/config/spec/environment/model/population hashes.
5. ONLY in a session explicitly authorized to execute: sbatch one MIG extraction+Exp1 job.
   Extraction collects only B baseline states; seal cross-fit directions before any pulse outcomes.
   Real-model engineering preflight verifies OLD no-edit/alpha0 identity and OLD historical match,
   D2 scratch value identity/cache isolation. A failure invalidates attempt; no analysis/selection.
6. Run all three Exp1 candidates/no-edit on180 positions at exact e*. Analyze then independent
   audit. Neither new candidate qualifies -> stop; validity/audit failure -> stop. No repopulation.
7. If selected/audited, seal selection artifact and Exp2 manifest; run only no-edit/OLD/selected
   gate-coupled replay. Analyze/audit; any gate-coupled benefit/safety failure -> stop before Exp3.
8. If audited pass, seal Exp3 manifest; run B0/OLD/NEW/BROAD/AUTO on all300 utterances. BROAD
   uses sealed NEW energy budget and B0 count rule; reference-free decoders never read evaluator
   data. Evaluate canonical metrics and independent audit. No P3 automatically authorized.
9. Commit reports, immutable scalar results/manifests/audits, source and config hashes; push.
   Large tensor artifacts stay local content-addressed with paths/hashes. Report failed attempts.

## Stopping and reporting rules

No scientific continuation after absent/non-PASS audit, invalid cache/state identity, failed
population/energy/firewall/manifest checks, unsupported rank/sign, compute allocation failure,
or prior experiment gate failure. D1 degeneracy becomes invalid folds, no rank tuning; candidate
validity below95% stops qualification. D2 fallback remains zero edit, no alternate direction.
Poor outcomes are final, not grounds to rerun or change criteria. Mechanical defects may be
corrected with regression tests/new commit/new manifest and preserved invalid attempts.
No scientific conclusion without `P2_DIR_AUDIT: PASS`. Exact statistical gates and precedence
are spec sections6-9 and JSON; no implementation discretion to weaken them.

Expected run matrix: extraction (no outcome), Exp1 NONE/D0/D1/D2; conditional Exp2
NONE/D0/selected; conditional Exp3 B0/OLD/NEW/BROAD/AUTO. No companions, random reruns,
minus-old candidate, rank/layer/gate/dose sweep. Budget<=4 GPU-hours target, max2 jobs active.

## Design review checks performed

CPU-only source/artifact provenance inspection, JSON parsing/source hashes, frozen-population
counts/fold feasibility, full diff/whitespace review, existing focused regression suites.
No new scientific actuator/test implementation and no GPU experiment in this design session.

Verification environment: acl1 Python with `PYTHONPATH="$PWD/src:$PWD"`,
`LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` (same conda runtime path as Slurm).
Existing suites `test_inference_cf_p1`, `cached`, `ce`, `p2`, `p2r`, `p2rj` and
`test_dg02_site` passed CPU-only. Initial invocations without these paths failed
import resolution/CXXABI_1.3.15; no repository code or environment packages were changed.
A joblib shared-memory warning caused serial operation only. JSON/source hashes,
180 position/parent hashes and all-fold observation-count guards passed. Numerical
rank/sign guards still require later approved extraction; no result is implied here.
