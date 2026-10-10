# P2-DIR pre-run audit (before any P2-DIR scientific outcome)

Date 2026-10-06. Stage: P2-DIR direction identification (frozen spec
`P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md`, config `configs/inference_cf/p2_dir_direction_identification.json`,
handoff `P2_DIR_CODEX_DESIGN.md`, all frozen at `3ef6542` and byte-unchanged here).
Machine verdict file: `results/inference_cf/p2dir/prerun_audit.json` (audited implementation commit `1802441`)
(independent auditor `experiments/inference_cf_p2dir_audit.py prerun`).

**Verdict: `PASS_TO_P2_DIR_EXP1`** (the spec's `PASS_TO_P2_DIR_RUN` label, scoped to Exp-1).

## Start state

Local root `/home/tungnx/cs-asr-steer-inf`, branch `feature/inference-cf-steering`; local HEAD and fetched
`origin/cs-asr-steer-inf` both `3ef654246c05307a6cddd5fe706c089d696cbbee`; clean tree; empty Slurm queue.
No pre-existing local work; nothing restored, reset, merged or force-pushed.

## What was implemented (exactly three directions)

| Item | File | Contract |
|---|---|---|
| Common interface | `src/csasr/inference_cf/directions.py` | `DirectionContext` (h_b, h_e, pre-step B cache, encoder output, current tokens, start, step — no reference field) -> `DirectionResult{id, direction, status, reason, provenance, counters}`; no fall-through between providers; dose/energy owned by the caller |
| D0 OLD | same, `OldDirection` | delegates `core_p1.direction` unchanged (epsilon-normalized, not renormalized) |
| D1 UNIQUE | `src/csasr/inference_cf/unique.py` | `NEW_P2_DIR_CROSSFIT_V1`: leave-one-dialogue-out, A = EN-correct, B = ZH-correct baseline B states, uncentered float64 second moments, `eigh` top-32 exactly, floor/gap/ortho/pair/score/sigma/sign guards at the frozen tolerances, sign toward mu_A - mu_B, float32 unit vector; invalid fold = zero edit, no fallback, no global refit |
| D2 READOUT | `src/csasr/inference_cf/readout.py` | J = log(P_E+1e-12) - log(P_M+1e-12) on suppression-processed float32 logits with the frozen `core_r2.tokenizer_partition`; float32 zero leaf probe at u_source of the current query; one `autograd.grad`; float64 tangent + epsilon normalization; unit/radial guards; pre-step cache deep-copied into ordinary tensors under `inference_mode(False)`; parameter flags restored, no parameter grads; persistent cache fingerprint unchanged; no P2-RJ import |
| BROAD budget (Exp-3, pre-committed) | `src/csasr/inference_cf/broad.py` | equal squared-energy packets sqrt(Q_u/N_u), last packet capped, actual-energy decrement, clamp, exhaustion, 2% / 5% validity |
| Runner | `experiments/inference_cf_p2dir.py` | `extract` (B/E states only), `exp1` (seal D0/D1/D2 then fixed-order pulses at e* via the unchanged P2-R `pulse_hook`/`solve_scale`, crop + bitwise restore), `exp1-eval` (evaluator Jacobian; refused until directions are sealed) |
| Preparation | `experiments/inference_cf_p2dir_prepare.py` | construction projection (field allowlist), D1 folds from extraction states, per-stage immutable manifests (clean Git, env incl. NumPy/LAPACK config, model file hashes, partition/suppression hashes) |
| Analysis | `experiments/inference_cf_p2dir_analyze.py` | evaluator metrics from saved lossless logits; family-10 shared-draw dialogue bootstrap; frozen qualification/selection; Exp-2/Exp-3 decision rules as pure functions (frozen now) |
| Independent audit | `experiments/inference_cf_p2dir_audit.py` | no import of the analysis module; `prerun`, GPU `spot` (independent D2 recomputation, 10/stratum by sha256 order), `exp1` (fold refit, D0/D2 provenance, energy from saved states, metrics, bootstrap, labels) |
| Slurm | `slurm/inference_cf_p2dir.sbatch` | MIG 3g.40gb, `STAGE=extract|exp1`; exp1 requires this PASS file |

No existing frozen primitive was modified (`core_p1`, `core_r2`, `sites.py`, `hooks.py`, `inference_cf_cached.py`,
`inference_cf_p2r.py`, `inference_cf_p2rj.py` are byte-identical to the design anchors). Population is the
byte-identical 180-position P2-RJ file (positions hash `sha256:c1e016c0…`, parent P2-R population
`sha256:6a880d2b…`); construction projection `results/inference_cf/p2dir/construction_population.json`
holds only `utterance_id, dialogue_id, t, stratum` and the 20 sealed leave-one-dialogue-out fold plans
(each group 55-59 observations >= 32).

## Tests (CPU)

`tests/test_inference_cf_p2dir_directions.py` (27), `..._protocol.py` (10), `..._audit.py` (16); plus the
unchanged existing suites `test_inference_cf_{p1,cached,ce,p2,p2r,p2rj,p2rje}` and `test_dg02_site`.
Coverage per spec section 10: D0 bitwise == `core_p1.direction` including tiny/nonfinite; D0 Exp-1 arm
== historical P2-R `+d` pulse (bitwise solver s, edit norm, cos_d_state) on a bf16 tiny Whisper; D1
synthetic known subspace recovery, energy score, top-32 boundary tie, count/rank/score/sigma/sign guards,
sign flip, deterministic serialization/hash, fold exclusion of every utterance of k, no global refit; D2
finite-difference gradient, suppression/epsilon, tangent/unit, tiny/nonfinite no-op, parameter flags and
grads, cache fingerprint, scratch logits/site bitwise == clean B step, outer inference-mode, future-token
independence, exception cleanup, AST firewall; runner sealing, matched energy (target and pairwise 2%),
zero-dose bitwise identity, restore, lossless bf16 logits, manifest/firewall rejection; hand-computed
bootstrap, omitted draws, determinism, every threshold equality, Exp-1/2/3 stop and precedence cases,
BROAD exhaustion/early EOS/mismatch; auditor/analysis agreement (<= 1e-12) and independence.

## Implementation interpretations (recorded, not scientific choices)

1. **Corruption** at a correct-stratum position = post-edit decode-rule (first-index) argmax not in the
   frozen Y_ref; `top1_changed` co-reported. **Correction** at a confusion position = argmax in Y_ref.
2. **Observed per-stratum corruption rate** = the dialogue-weighted point estimate (same aggregation as
   every statistic); pointwise 95% per-stratum CIs co-reported.
3. **Bootstrap draw universe** = the 20 sorted dialogue IDs of the frozen population (outcome-independent);
   one draw matrix `default_rng(240924).integers(0, 20, (10000, 20))` shared by all statistics; a
   statistic's draw is omitted (and counted) when none of its drawn dialogues has rows; percentiles by
   `numpy.quantile` (linear).
4. **Valid rate denominator** = all 180 frozen positions; an arm is valid at a position only if the
   historical state identity holds and direction, solver, hook, and energy checks pass. Run-level
   engineering (-> `P2_DIR_INVALID`): completeness, sealing, bitwise checks (extraction, D2 scratch
   logits/site, restore, evaluator logits/site), <= 9 state-identity mismatches, D0 valid >= 95%,
   >= 9900 valid draws. Candidate-level (-> no qualification): candidate valid < 95%, matched population
   < 40 positions or < 12 dialogues in any stratum.
5. **Two GPU allocations** (extract, then exp1) with the CPU D1 fold construction in between, instead of
   one combined job; both are within the 1 h / <= 2 active job budget and no outcome exists before the
   folds are sealed.
6. **Evaluator Jacobian** runs as a separate post-seal pass (`exp1-eval`) using the P2-RJ zero probe
   (diagnostic module, imported lazily only there); it can never reach a direction.
7. **Audit tolerances**: D1 refit and D2 recomputation <= 2e-6 elementwise (construction bound); D0
   independent float64 recomputation <= 1e-6; recorded J vs float64 recomputation <= 1e-4 (float32
   objective), GPU spot J <= 1e-5; realized energy from saved before/after (recorder) states within the
   frozen 2% bound and within 5e-3 relative of the hook-reported norm (bf16 rounding, P2-R max 1e-3).
8. **Raw logits** are stored losslessly as bf16 bits per utterance, kept local (`.gitignore`), with sha256 in
   the committed row JSONs and the seal file; D1 fold eigen-arrays likewise local with hashes. All compact
   rows, vectors, folds, manifests, analyses and audits are committed.
9. **Exp-2 / Exp-3 runners** are NOT implemented yet. Their decision rules, family definitions, thresholds,
   precedence and the BROAD energy rule are frozen in code and tested now. The runners will be written
   only if Exp-1 selects a direction, reusing the P2-R replay / cached decode exactly, with their own
   tests, manifest and pre-run check committed before any Exp-2/Exp-3 outcome.

## Firewall

Only already-exposed D-dev-select P2-R/P2-RJ positions (80 utterances, 20 dialogues) and the R2/P2
300-utterance panel (Exp-3, conditional). No router-calib, D-dev-confirm, D-test, P3 or transfer corpus.
P3 HELD.
