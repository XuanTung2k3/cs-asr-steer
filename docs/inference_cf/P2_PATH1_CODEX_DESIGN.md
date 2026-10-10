# P2-PATH1 — implementation handoff

Design-only freeze. No PATH1 runner/outcomes implemented or run. Spec, config and fingerprinted sites are normative. Historical PATH0/A4 code/config/reports/results remain immutable; core v6 unchanged.

## Reuse map

- `src/csasr/inference_cf/path_decode.py`: unchanged `clamp_decode` fresh `cached.Branch`, stream/ED/allowed_at and branch diagnostics. It records prefix/suppression/position failures rather than universally raising: the new runner **must** inspect all returned trace checks and reject any failure. Same no-clamp decode path/passive16 attention and generation settings.
- `experiments/inference_cf_p2path0.py`: reuse prepare/manifest/audio/model/encoder/reconstruction/reset pattern and12-row barrier. Do not run the old scientific L1/L3 matrix again. PATH1 all12 reconstruction is integrity+score-state preparation; only U receives four new L1 factorial paths.
- `script_safe_tta.adapt_a4`, `soft_auto_tta`, `episodic_tta.LNGuard`: byte-identical A4 teacher/194 masters/2 updates/fresh optimizer/precision/materialization/reset. Historical fp32 checkpoint cast bf16 is comparator only. Encoder clone outside inference_mode must match original; no encoder grads. Use same sealed native AUTO condition (historical detect_language input_features path) and check equality.
- PATH0 original `primary_analysis.json`/output seal/primary and post audits: frozen A4A old outputs, suffix denominators and criterion. Recompute them independently; old verdict alone does not replace per-token checks. Branch diagnostics may be reused with source/state/site/generation provenance.
- Parent PATH0 `SITES.json` and `plan_sealed.json`: exact common prefixes, teacher masks/classes/audio/lang/checkpoints. New sites fingerprint parent rows, not new science. Parent source B0/AUTO/A4 arrays remain the only donor sources.
- `experiments/inference_cf_p2path0_analyze.py::row_endpoint/criterion`: primary implementation may reuse exact inherited induction arithmetic. Independent auditor must implement its own ED/criterion/labels and must not import this primary decision code. Canonical reference evaluators/P2SEQ utt_counts can be shared.

## Additive implementation files

Add `src/csasr/inference_cf/branch_adjudication.py` for short donor log-prob scoring and state-owned path execution; no gradients/new objective. Use cached.Branch sequential prompt/common-prefix/donor-token feeding, not teacher_logits' full-prefix uncached path. Teacher forcing here means supplied donor tokens, **not** another loss/update.

Add `experiments/inference_cf_p2path1.py` (prepare/manifest/run/seal), `_analyze.py` (reference-free factorial/state/score decision and separate guarded reference command), `_audit.py` (independent pre/primary/full), `slurm/inference_cf_p2path1.sbatch`, `tests/test_inference_cf_p2path1.py`. No modifications to historical A2/A3/A4/PATH0 modules required. Resolved runtime code hashes are recorded in the new manifest; parent source hashes must remain byte-identical.

Outputs `results/inference_cf/p2path1/run1`; plan/preaudit/primary seal and analysis at `results/inference_cf/p2path1/`. Later report `docs/inference_cf/P2_PATH1_REPORT.md`; no report created now.

## State/cache ownership contract

Select/materialize model state **before** creating a Branch. Set a distinct owner `{state_hash, row, condition}`; cache starts None, fed/positions empty. Keep that exact resident state immutable until the path ends. Never pass a decoder cache to another owner or switch LN state while a branch is live. Destroy all branch/cache objects before `guard.restore`/next materialization. Frozen encoder tensor may be shared because decoder-LN updates do not alter it.

Record resident LN hash before/after each path, fresh-cache flag, full consumed content/prompt order or compact reproducible trace, contiguous positions/length and owner identity. Independent auditor inspects implementation plus trace/hash consistency; equal cache lengths alone cannot prove correct ownership. Use scoped try/finally materialization/cache disposal/reset. Model×branch B and ALT must also have distinct fresh caches even when their prefixes match.

## Run sequence

1. CPU prepare independently verifies new sites against parent raw sources/termination: exact U7/C5, k/prefix/c_B/c_A, six eligible/one direct, H=3 donor content lengths, source hashes. Hash-check PATH0 primary seal/audits and original INDUCE1. Push implementation/tests/plan; independent **PASS_TO_P2_PATH1** then resolved manifest pushed.
2. One30-minute sbatch allocation. Reconstruct each of12 A4 states once through original teacher/updates. Theta0 replay=B0 and FREE-A4 replay=sealed output, effective checkpoint and resets all12 before intervention/scoring barrier. Capture/store tiny states and detached encoders; no alternate state on mismatch.
3. Seven U: all four new T0B/T0A/A4B/A4A decodes, fresh cache under selected state throughout. Only force c_B or c_A once. Immediately assert T0B/A4B match B0, A4A matches old PATH0 L1 tokens AND termination. All trace/state/reset failures invalid. No L3 run.
4. All12: four fresh short score paths per row (two models×two donors). Process logits at absolute content index k+j; log-prob donor before consuming it. Common prefix replay sequential under selected state; no greedy completion or EOS target score. H_eff separate per branch, EOS excluded, first-position suppression depends on k+j not donor-relative j. No parameter/backward updates.
5. Compute reference-free suffix metrics, inherited pass, state mechanism label, scores/ties/CONSENSUS_REJECTS_AUTO. A4A must reproduce parent induction PASS; any failure invalid. Seal all factorial/score outputs, checks/traces, primary inputs/label and hashes; commit/push before any reference loading.
6. Independent primary audit before opening references; secondary U/C canonical evaluation/ASR flag behind committed seal. Independent **P2_PATH1_AUDIT: PASS**, commit/push report and STOP. No post-outcome score replacement, threshold/horizon changes, gate deployment or subsequent TTA.

## Focused tests

- Parent/new site bytes and row fingerprints; exact U7/C5, unchanged first divergences/common hashes/candidates; EOS/cap/content indexing; suffix denominator identity and DIRECT_ONLY handling.
- Original A4/optimizer194 names/reset/precision regression; all-row barrier and effective checkpoint identity. All new T0B/A4B identities and A4A exact old token/termination reproduction are mandatory runtime tests.
- Fresh model×branch cache ownership: deliberately stale cross-state cache fixture rejected; cache created after state selection; no state mutation until disposal; no prefix prefill shortcut; encoder sharing allowed; contiguous cache positions and correct consumption of forced token.
- Same greedy/suppression/cap semantics; clamp length exactly1. Trace false cannot be ignored. No new teacher/optimizer settings or steering/D2.
- Exact inherited INDUCE1 criterion copied, suffix excludes forced span; signed I/Delta, pooled count sums, tau=max(0.25,1/d0); strict vs inclusive boundary/tolerance tests. Exhaustive mechanism precedence, heterogeneous/disjoint-row fixtures and insufficient-evidence NO_CLEAR.
- Score exact per-token conditional log probabilities, absolute step suppression, H=3 content and shortened near-EOS branches, no EOS scoring/control substitutions; mean length normalization, 0.5 consensus, tie B at1e-12, S_min descriptive only. Strict-win counts versus tie selections and eligible-only criterion.
- No reference/evaluator access before primary seal; primary label invariant to secondary metrics; ASR_STATE_ALIGNMENT exact count rules and empty-population behavior. Auditor imports no primary decision logic.
- One job/12 states/28 factorial paths/48 short score paths,24 updates/≤25 backwards; no L3, extra population, fallback or sweep. Partial/budget failure INVALID STOP.
