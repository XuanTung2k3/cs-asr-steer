# P2-DIR — current-state verification and Claude handoff

**Status: POST-OUTCOME VERIFICATION, 2026-10-06.** This document introduces no scientific
choice, pre-registration, actuator, run or threshold revision. P2-DIR was frozen at
`3ef654246c05307a6cddd5fe706c089d696cbbee`, implemented at `1802441`, and has already
completed Exp-1. Its terminal result is **P2_DIR_NO_NEW_DIRECTION_SUPPORTED**, with
supplementary **P2_DIR_CAUSAL_POWER_WITH_DAMAGE:D2**. Exp-2/3 are blocked by the original
selection rule. P3 remains held. Do not interpret the historical config's
`DESIGN_FROZEN_IMPLEMENTATION_PENDING` label as the current execution status.

Authoritative unchanged pre-outcome documents:

- `docs/inference_cf/P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md`
- `configs/inference_cf/p2_dir_direction_identification.json`
- `docs/inference_cf/P2_DIR_CODEX_DESIGN.md`

Implementation clarifications were recorded before outcomes in `P2_DIR_PRE_RUN_AUDIT.md`.
Results: `P2_DIR_EXP1_REPORT.md`, `P2_DIR_FINAL_REPORT.md`, and the existing immutable
`results/inference_cf/p2dir/` artifacts. This handoff resolves the repeated design request
against actual newer local state; it does not recreate a pre-outcome freeze after seeing results.

## Exact starting state and preserved local work

Root `/home/tungnx/cs-asr-steer-inf`; branch `feature/inference-cf-steering`.
Local HEAD and fetched origin/cs-asr-steer-inf both
`c899fa1362ab3ac0aebcf0b6a20f8d2cceb225b8`; no ahead/behind commits, no tracked diff;
empty Slurm queue. No reset, branch change, merge, force-push or PR.

Two pre-existing untracked files were found before this verification:

| File | Starting SHA256 |
|---|---|
| `configs/inference_cf/p2_sel.json` | `f85e7a1b795f48f07e4c404611d4cbba6cb446805f21da156af2f70fc30a4eb0` |
| `docs/inference_cf/P2_SEL_MINI_PANEL.json` | `55bfa0ece57584a730b8b96006c61f58f5086694d97dd8a7cf8f098b34006461` |

They are P2-SEL drafts outside this latest P2-DIR request. Git provides no author/commit provenance.
The config refers to a P2_SEL_SPEC.md that is absent, so it is not a complete reviewable scientific
freeze. Both files are preserved byte-for-byte, unstaged and uncommitted; no deletion or restoration.
They must not authorize a run or bypass P2-DIR's stop. All deliverables of this P2-DIR verification
are versioned; these unrelated incoming drafts are explicitly disclosed, not claimed as clean state.

## Actual current implementation

| Component | Current path and status |
|---|---|
| R2 gate/localizer/partition | `src/csasr/inference_cf/core_r2.py`; unchanged E, R_B, max-attention 1s window, script partition |
| OLD math | `src/csasr/inference_cf/core_p1.py:direction`; unchanged |
| DG-02 site/repair | `src/csasr/lss/sites.py:DecoderPostCrossAttnInterventionHook` -> `models/hooks.py:apply_steering`; unchanged |
| Cached free decoding | `experiments/inference_cf_cached.py:Branch,cached_decode`; unchanged OLD B/E/S execution; no new-provider integration yet |
| Historical pulses/replay | `experiments/inference_cf_p2r.py:DiagBranch,pulse_hook,solve_scale,replay`; unchanged |
| Evaluator Jacobian | `experiments/inference_cf_p2rj.py`; imported lazily only for post-seal evaluation, never by providers |
| Final diagnostic enumeration | `experiments/inference_cf_p2rje.py`; diagnostic programme ended |
| Common provider | `src/csasr/inference_cf/directions.py:DirectionContext,DirectionResult,OldDirection,UniqueDirection,ReadoutDirection`; implemented |
| D1 construction | `src/csasr/inference_cf/unique.py`; implemented, 20 sealed cross-fit folds valid |
| D2 construction | `src/csasr/inference_cf/readout.py`; implemented reference-free scratch forward |
| BROAD budget | `src/csasr/inference_cf/broad.py:BroadBudget`; implemented/tested pure budgeting rule, no free-decoding run |
| Preparation | `experiments/inference_cf_p2dir_prepare.py`; construction, folds, manifests |
| Runner | `experiments/inference_cf_p2dir.py`; extract, exp1, exp1-eval only |
| Analysis | `experiments/inference_cf_p2dir_analyze.py`; Exp-1 metrics/bootstrap/selection plus pure Exp-2/3 decision functions |
| Independent audit | `experiments/inference_cf_p2dir_audit.py`; prerun, spot, exp1, final; independent of analysis |
| Launcher | `slurm/inference_cf_p2dir.sbatch`; extract/exp1 with manifests/pre-run audit required |
| Canonical transcript evaluation | `experiments/inference_cf_p2_evaluate.py`, `src/csasr/evaluation/{canonical,retention}.py` |
| Provenance infrastructure | `csasr.utils.provenance`, `csasr.lss.specfreeze`, inference_cf manifests/hashing |

Exp-2/3 runners and corresponding independent post-run audit execution are intentionally absent:
they were conditional on selection, and selection failed. Do not implement or launch them as
routine completion of P2-DIR. The current source matches the pre-run implementation descriptions.

## Recovered unique/shared provenance

Historical numerical code: `src/csasr/experiments/basis_a5_unique_shared.py`, last commit
`e00385d306717d21f19e294ce2c922ff44f80538`. Extraction:
`experiments/basis_a5_unique_shared.py:construct_whisper`. The surviving A5 manifest records
provenance commit `2b307645b05f91b080302d8f33f755212446dca1` (manifest refresh).

| Field | Historical A5 versus frozen P2-DIR choice |
|---|---|
| A population | A5 D-construct English tokens in correct eligible spans; P2-DIR frozen EN-correct B-branch diagnostic states |
| B population | A5 preceding token positions, not explicitly filtered to Mandarin correctness; P2-DIR frozen ZH-correct B-branch states |
| Site | L16 post-cross-attention/pre-FFN in both |
| Timing | A5 reference teacher-forced query prefix_length+i-1; P2-DIR unedited historical baseline-prefix query predicting t |
| Centering | Uncentered in both |
| Rank | A5 top64 bases with selection among first32 principal modes; P2-DIR exactly top32 for both groups |
| Math | U_A^T U_B=P Sigma Q^T; a_i=U_A p_i; E_A=a_i^T C_A a_i; maximize E_A(1-sigma_i^2) |
| Sign | Positive dot with mu_A-minus-mu_B; P2-DIR rejects an indeterminate anchor |
| Leakage | Historical D-construct/evaluation separation but exposed oracle-local results; P2-DIR leave-one-dialogue-out throughout, conditional development CIs |
| Reuse | A5 original file path absent; recovered A6 fixed-CS copy is hash-verified but excluded from P2-DIR because its construction semantics differ |

Recovered vector: `results/basis_a6_expanded/fixed/cs_dialogue/directions/whisper/decoder/L16/add_unique.npy`;
file hash `93d681722d8831460ef241e27a77377036d7dc179be9e13893ffe1be376e9830`;
float32 .npy reserialization matches A5 hash
`f82f7790b6df3ee2a9b8cacda1a98c91a35cdb7e894008e58dde15314f9f7bff`.
A5 L16 CS Add-Unique corrections/corruptions 60/19 versus Minus-Shared 21/247 at unequal energy;
these exposed oracle-local results are descriptive only. A5 rank/population discrepancies remain
explicit historical IMPLEMENTATION GAPs in METHOD_CONTRACT, with no historical repair or relabeling.
A-unique remains a representation hypothesis, not a pure language-specific axis.

## Unchanged three-direction contract

D0 delegates core_p1.direction: same-prefix forced-EN versus forced-ZH exact-site states,
float64 subtraction/norm, delta/(norm+1e-6) to float32; norm<1e-4 or nonfinite -> no edit.
B/E caches independent, only shared currently generated content; S has independent edited history.
Deployment does not renormalize OLD; diagnostic chord solver has inherited unit normalization.

D1 NEW_P2_DIR_CROSSFIT_V1: EN-correct A and ZH-correct B exclude every position of evaluated
dialogue. Uncentered uniform position moments in float64, exactly32 eigenvectors per group,
NumPy eigh then SVD of cross-basis matrix, deterministic score/sign and degeneracy guards.
Normalize float64, serialize float32; invalid fold -> no alternate vector/rank/global refit.
Runtime receives only a sealed vector. Twenty folds have been constructed and audited.

D2 p2dir_readout_v1: J=log(P_E+1e-12)-log(P_M+1e-12), frozen whisper_han_ascii_v1 partition,
suppression-processed float32 logits, temperature1. Differentiate at unedited B L16 site through
one float32 zero leaf cast to bf16, autograd.grad once. Scratch pre-step cache/encoder tensors
are ordinary detached non-inference copies under inference_mode(False); parameter flags restored,
parameter grads absent, persistent cache intact. Tangent/unit geometry float64 then float32 output;
tiny/nonfinite/normalization failure -> no edit. Evaluator Jacobian/reference targets never enter
construction or orientation. Evaluator diagnostics are strictly post-seal.

## Frozen run matrix, thresholds and current stopping state

Exp-1: all180 original positions (60/60/60), same B state/prefix, L16 DG-02 NormPreserve;
NONE plus D0/D1/D2 independent single pulses at e*=1.1260757575454359. Actual squared-energy
mismatch versus target and between arms <=2%, unchanged P2-R solver max8 evaluations.
Primary fixed-competitor reference margin; reference logp, script masses, rank, correction,
useful-gradient alignment/kappa, chord/relative edit and correct-state margin/corruption retained.

Statistic: mean positions per dialogue then equal dialogue means; shared20-dialogue draw universe,
10000 paired draws/default_rng(240924); percentile Bonferroni family10. Candidate qualifies iff
confusion point>=.5 nat and lower>0, paired-versusD0 lower>0, EN/ZH correct margin lower>=-.25,
pooled correct corruption upper<=.05 and per-language observed corruption<=.05, engineering valid.
Both qualify: D2 only if paired D2-minusD1 pointwise95% lower>.25 nat; else cheaper static D1.
Neither qualifies: stop. No outcome changes these gates.

| Actual Exp-1 | D0 | D1 | D2 |
|---|---|---|---|
| EN-confusion margin | -.020 nat | +.227 [.079,.421] | +4.499 [3.829,5.168] |
| ZH-correct margin | -.047 | -.040 [-.099,.017] | -2.011 [-2.466,-1.584] |
| Confusion kappa | -.011 | +.021 | +.797 |
| Qualification | Historical comparator | Materiality fails | Safety fails |

All180 states/three arms valid, D0 historical regression reproduced, D2 independent GPU spot30/30,
20/20 D1 folds valid. D2 ZH-correct corruption .137 (8/60); no selected new direction.

Exp-2 only if selected: persistent baseline-prefix NONE/OLD/NEW replay, unchanged gate E*R_B,
alpha2/phi_id/localizer/L16/NormPreserve. Benefit point>=.10 nat AND>=20% Exp-1 benefit, lower>0,
paired-versusOLD lower>0; unchanged safety. Family5. **Currently NOT RUN / BLOCKED.**

Exp-3 only if audited Exp-2 pass: same exposed300/20-dialogue panel, B0/OLD/NEW/BROAD/AUTO.
BROAD sealed per-utterance NEW squared-energy budget, equal packets Q/N using B0 eligible count,
actual-energy decrement, final cap/exhaustion,2% aggregate and5% utterance-mismatch validity.
Canonical metrics/retention/harm and runtime/backward/VRAM; family10, material PIER gain>=.005
and lower>0 versus bothB0/OLD, MER/ZH-CER damage upper<=.005, EN-WER<=.01, retention loss<=.01,
outside-harm<=.005. **Currently NOT RUN / BLOCKED.**

Decision tree unchanged: invalid engineering/audit blocks inference; no qualifier stops;
local benefit plus damage gives supplementary causal-power-with-damage; conditional Exp-2 benefit
failure suspects gate coupling; conditional safe BROAD superiority suspects selective gate;
conditional safe material NEW superiority supports OLD direction bottleneck; remaining local-success/
free-decode-failure suspects site or sequence leverage without proving site failure.

## Verification performed in this session

No GPU inference or new model-output construction. All16 frozen source/artifact SHA256 anchors
match current files. Spec/config/original handoff match freeze commit3ef6542 byte-for-byte.
Existing independent `cmd_final` was run CPU-only to a new scratch output, preserving archived audits:
P2_DIR_AUDIT: PASS; terminal label P2_DIR_NO_NEW_DIRECTION_SUPPORTED; every check true.
The new verification JSON is `docs/inference_cf/P2_DIR_CURRENT_STATE_CHECK.json`.
This session-level recheck verifies archived audits and stop/provenance checks; it is not a fresh
full reconstruction of Exp-1 gradients/statistics. The existing Exp-1 audit remains the source
for that reconstruction. No scientific conclusion was changed.

Existing focused suites passed: **55 tests** in7.39s (directions/protocol/audit). CPU-only
acl1 environment with conda libstdc++ path, repository PYTHONPATH and OMP/MKL threads1.
Joblib emitted a shared-memory warning and used serial operation; all tests passed.
Coverage includes D0 historical identity; D1 principal-angle/rank/sign/orthonormality/exclusion/
serialization; D2 objective/tangent/autograd/grads/cache/future-token/fallback/exception cleanup;
matching/zero-edit/sealing/firewall/manifests; bootstrap/threshold/selection/stop/BROAD/audit independence.
No new tests were needed for a documentary current-state clarification.

## Claude exact action sequence

1. Inspect actual HEAD/status/remote; preserve all newer local work and the incoming untracked drafts.
2. Read this current-state handoff, unchanged original spec/config, pre-run audit, Exp-1/final reports.
3. Reuse completed providers, extraction, folds, Exp-1 artifacts and audited analysis. Do not recreate
   a pre-outcome contract, rerun a valid unfavorable comparison or relax a safety/materiality threshold.
4. Apply selection exactly: selected=None -> stop. Do not build/submit Exp-2 or Exp-3 under P2-DIR.
5. Do not interpret incomplete P2-SEL drafts or a historical pending status as run authorization.
   Any separately authorized next-stage design must have its own complete pre-outcome freeze and
   acknowledge exposure to P2-DIR outcomes; it cannot be relabeled P2-DIR continuation.
6. Every reviewed authorized edit: check, inspect diff, commit, push origin HEAD:cs-asr-steer-inf.
   Preserve legacy outputs; no force-push, main merge or PR.

Current P2-DIR files needing scientific modification/addition: NONE. Files to reuse are listed above.
Forbidden modifications: frozen spec/config/thresholds/population, archived results/audits, historical
A5/A6 artifacts, OLD/R2/site/repair math, data splits. Firewalls remain D-dev-select/historical
artifacts only; no router-calib/new validation/D-dev-confirm/D-test/P3/transfer data.
Original compute contract remains H100/MIG/sbatch/max2 active jobs/few GPU-hours, but the
current P2-DIR stopping rule permits **zero additional scientific GPU jobs**.
