# Inference-time counterfactual steering: implementation guide

**Project folder:** `cs_asr_steer_inf`  
**Status:** P0/P0-R1 completed; original geometry/candidate-support gate is blocked. The unsteered P0-R2 repairability gate is specified in `docs/inference_cf/P0_R2_REPAIRABILITY_SPEC.md` (v1.1), implemented, and independently audited `PASS_TO_R2_RUN` (`docs/inference_cf/P0_R2_REPAIRABILITY_AUDIT.md`). **R2 ran (Slurm 54758, manifest `sha256:e8869f0c…`, commit `fe960e4`): `R2_CF_FEASIBLE_NOT_PREFERRED`; `POST_R2_AUDIT: PASS`; selected gate for P1 `g_selected = g_old = E_t R_t^B`** (`docs/inference_cf/P0_R2_REPORT.md`, `P0_R2_POST_RUN_AUDIT.md`). `g_cf` was feasible (AUROC 0.710/0.726) but significantly worse than `g_old` (0.820/0.836; paired ΔAUROC −0.11, CI excludes 0) because the same-prefix English counterfactual barely moves Han mass (median `D≈0` at confusion positions). **P1 done: `P1_CAUSAL_ACCEPTANCE_PASS`, `POST_P1_AUDIT: PASS`** (spec `docs/inference_cf/P1_CAUSAL_ACCEPTANCE_SPEC.md` v1.1, job 54781, manifest `sha256:6e595613…`, commit `d704db9`; attempt 1 job 54770 `P1_BLOCKED` by an A6 instrument defect, preserved): gate `g_old`, direction `norm(hE−hB)`, decoder L24 post-cross-attention/pre-FFN site, NormPreserve, α∈{0,1.0} engineering values; bitwise α=0 identity, 20 real edits, isolation verified (`docs/inference_cf/P1_REPORT.md`, `P1_POST_RUN_AUDIT.md`). *(Historical at P1 close: `READY_FOR_P2_COMPACT_DEVELOPMENT`; P1 claims computational validity only.)* **Pre-P2 COMPLETE (`CACHED_EQUIVALENCE: PASS`, job 54802). P2 COMPLETE: `P2_VALID_CONFIGURATION_SELECTED` → project state `READY_FOR_P3_LOCKED_EVALUATION`, frozen L16/α=2.0/φ_id/g=E·R_B/+d (`docs/inference_cf/P2_REPORT.md`); `OPEN DECISION` before P3 because the development effect is negligible and P2-B mechanism controls are negative. P3 not started.** **P2-R COMPLETE (job 54874, `P2_R_AUDIT: PASS`): `P2_R_MECHANISM_STILL_AMBIGUOUS` — at L16/P2 energy every tested direction is ~inert at the lexical decision and +d is the worst of +d/−d/random; oracle relocation gives no material rescue (`docs/inference_cf/P2_R_REPORT.md`). Project state `P2_COMPLETE_MECHANISM_UNRESOLVED`; P3 HELD.** **P2-RJ COMPLETE (job 54998, `P2_RJ_AUDIT: PASS`): `P2_RJ_STILL_AMBIGUOUS` — the evaluator-only reference-margin Jacobian at the same 180 states shows d=norm(hE−hB) captures ~−1% of the available first-order leverage (random-level alignment; H1 supported under every leverage class), while an optimally aligned e*-edit would move the margin by a median 5.1 of 10.6 nats (Q50 0.61 [0.44, 0.76], at the frozen 0.5 boundary; H4 co-occurrence unresolved) (`docs/inference_cf/P2_RJ_REPORT.md`). No repair frozen; P3 HELD.** **P2-RJ-E COMPLETE — DIAGNOSTIC PROGRAM ENDED (job 55002, `P2_RJ_E_AUDIT: PASS`): `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED` (terminal). On all 227 eligible EN-confusion positions (17 dialogues) Q50 = 0.539 [0.406, 0.665] → L16 leverage at e* unresolved; H1 confirmed (κ(+d) ≈ −1%). Next: one independent actuator-design pass from `docs/inference_cf/P2_RJ_E_ACTUATOR_REDESIGN_HANDOFF.md`; P3 HELD.** `g_old` is a repair-need ranking score, never proven repairability.  
**Source idea:** *Full Inference-Time Self-Localized Steering* (24 September 2026), revised after P0/P0-R1 feasibility findings.  
**Purpose:** A standing guide for a coding assistant to implement, verify, and evaluate an entirely inference-time, training-free code-switching ASR intervention.

**Paper-completeness audit:** Section 9 defines the required experiment matrix and completion checklist. It supersedes earlier minimum-run suggestions where they differ. Completion means enough evidence to write the declared paper scope, including negative results; it cannot guarantee positive findings, publication acceptance, or that a new discovery will never require a follow-up experiment.

## 1. Goal and boundaries

For audio `x` and the currently generated content prefix `y_<t>`, obtain a baseline decoder state `hB` and two language-conditioned counterfactual states `hM` and `hE`, where `M` is the matrix language and `E` the embedded language. The method separates two questions:

1. **How to repair:** use the local conditioned displacement `hE - hM` as a candidate steering direction.
2. **When repair is needed:** use a reference-free gate based on (i) local acoustic support for `E` and (ii) contradiction between that support and the ordinary decoder's current language preference.

The primary experiment remains Mandarin-English with Whisper-large-v3. A later multilingual/architecture expansion is an independent replication and must not retroactively tune the primary method.

**Important P0 finding for the selected Whisper baseline.** Matrix (`M`) and embedded (`E`) languages are supplied task metadata. The primary known-M baseline `B0` forces Chinese, so its ordinary condition equals explicit Mandarin conditioning (`cB == cM`) and `hB == hM`. This identity is specific to that condition. Ordinary **auto-LID** Whisper (`B0_AUTO`) is a mandatory unsteered comparator because the R2 audit found it stronger on development data. The earlier `rho/c` "matrix-side collapse" geometry is non-identifiable and retired. The P0/P0-R1 candidate-continuation evidence (`q_cross`, K=1 then K=3) is also retired as the main gate because conditioned branches usually produced the same discrete continuation. These negative feasibility results are preserved as motivation/evidence; they are not overwritten by P0-R2.

The **selected (frozen) repair-need detector**, after R2, is

```math
g_t = E_t R_t^B .
```

R2 also tested the counterfactual candidate `g_t^{cf} = E_t [R_t^B-R_t^{Ecf}]_+`, where `E_t` measures **positive English evidence beyond Whisper's null-template bias**, `R_t^B` is baseline excess matrix-script mass, and `R_t^{Ecf}` is the same mass under forced English with the **identical baseline content prefix**. This is an uncalibrated **repair-need ranking/strength** score. The previous `g_t^{old}=E_tR_t^B` was a predeclared R2 ablation. **R2 outcome: `g^{cf}` is feasible but not preferred; the frozen rule selects `g_selected = g^{old} = E_tR_t^B` for P1 onward** (the Ecf term is retained only as a reported negative diagnostic). The selected gate should be low on Mandarin and already-correct English and high on English-to-Han confusion; it does not target every failed English position.

**Training-free** means frozen model weights and no learned gate, probe, basis, adapter, or reference-derived switch detector in the deployable path. **Inference-time** means the direction and gate are computed from the current utterance/prefix without reference transcripts, gold future tokens, or aligned code-switch spans. Development examples may be used to validate/freeze architecture choices and hyperparameters; report this as *development-tuned, training-free inference*, not zero-shot hyperparameter selection. Always disclose extra model passes, local-LID cost, memory, and latency. Oracle/reference localization is diagnostic only.

## 2. Local folder and Git setup

The intended local checkout is named `cs_asr_steer_inf`. The actual repository and path must be verified on the machine where coding happens. Do not run these commands blindly if the folder already exists.

1. Inspect the original checkout: `AGENTS.md`, `CLAUDE.md` if present, Git branch/HEAD/status, current method contracts, code map, data exposure and result schemas. Inventory uncommitted A7 work separately; never reset or overwrite it.
2. Prefer a **new Git worktree** at `../cs_asr_steer_inf` on branch `feature/inference-cf-steering`, starting from a recorded committed source revision that has the validated decoder hook and evaluation machinery. For example, from the original repository: `git worktree add -b feature/inference-cf-steering ../cs_asr_steer_inf <verified-base-commit>`. If the folder already exists, inspect its Git identity and status first; do not overwrite it. If required A7 infrastructure exists only as uncommitted work, commit it appropriately in its source branch or port only the reviewed components in a traceable commit.
3. In the new checkout put the new method under a dedicated package/experiment namespace, following the repository's real structure. Suggested *relative* locations (adapt after inspection): `src/csasr/inference_cf/`, `experiments/inference_cf/`, `configs/inference_cf/`, `sbatch/inference_cf/`, `tests/inference_cf/`, `docs/inference_cf/`, `results/inference_cf/`. Do not duplicate the full legacy repository inside these folders.
4. Record source commit(s), copied files, modifications, and applicable licensing. Pin model revision, dependency versions, dataset/manifest hashes, decoder settings, and Git commit in run manifests. Keep large audio, weights, intermediate KV caches, and private data out of Git.

**Reuse candidate map (verify in code; paths are illustrative):**

| Existing capability | Reuse condition | New responsibility |
| --- | --- | --- |
| A6/A7 post-cross-attention/pre-FFN intervention and norm preservation | Exact state/hook semantics and beta=0 identity pass for current model revision | Runtime state and direction providers; do not retrofit an oracle mask |
| Standard Whisper decoding and beam wrapper | Confirm forced prefix, cache positions, beam lineage, token budget, and original baseline identity | Same-prefix conditioned branches and steered decoding |
| Canonical MER/PIER/EN-WER/matrix-CER, correction/damage, provenance | Match frozen definitions, reference normalization, evaluation populations | Gate/geometry/runtime fields and new result namespace |
| CS-Dialogue, ASCEND, SEAME adapters | Verify IDs, roles, split guards and local availability | Inference-only eligibility; separate manifests by data role |

## 3. Revised method contract and scientific gates

### 3.1 Historical P0 conclusions that are now frozen

The original feasibility sequence established:

- **Original G1 — condition geometry:** `cB == cM` for the primary Whisper Mandarin setup and `hB == hM`; therefore the previous midpoint projection `rho` and collapse factor `c` are algebraically degenerate and must not be used as a repair-need signal.
- **Original G2 — same-prefix validity:** the common-content-prefix, mask/cache, layer/site and greedy-lineage plumbing passed on the retained P0 panel.
- **Original G3 — continuation evidence:** the K=1 and K=3 branch-generated candidate contrast remained collision-dominated and did not satisfy the frozen support criteria. Do not continue a K sweep or silently redefine `q_cross`.
- P0 did **not** establish that `d = norm(hE-hM)` improves recognition. It established only that the conditioned displacement is finite/nontrivial and that the state-comparison plumbing is feasible.

These conclusions should remain visible in the audit trail and paper motivation.

### 3.2 Counterfactual steering direction: how to repair

At decoder layer `l` and logical token position `t`, hold audio and content prefix fixed:

```text
hB = F_l(x, y_<t; cB)
hM = F_l(x, y_<t; cM)
hE = F_l(x, y_<t; cE)
Delta = hE - hM
d = Delta / (||Delta||2 + eps)
```

For the selected primary forced-ZH Whisper baseline `hB == hM`, but keep the symbols distinct in the general interface because this need not hold for auto-LID, Qwen, or another architecture. A zero/tiny `||Delta||` must have a documented no-edit fallback.

### 3.3 P0-R2 hypothesis: local embedded-language acoustic support

The deployable gate needs a dense, continuous signal that asks whether the **local speech** supports `E`. Define generically

```math
E_t = LocalSupport(E | x,t;M,E), \qquad E_t \in [0,1].
```

The first Whisper Mandarin-English instantiation uses native local LID as a **model-specific evidence provider**. Obtain `x_t` causally from the mean of the shipped alignment heads' cross-attention at the **current query predicting `y_t`**, using only `(x,cB,y_<t)`. Mask to the heard encoder frames, renormalize, and choose the earliest **1.0 s / 50-frame window with maximum total attention mass**. The former point-argmax-centered 1.0 s window is a localization diagnostic. Zero-pad the selected waveform crop to Whisper's 30 s input. Post-hoc Whisper DTW timestamps and evaluator-only CTC timings cannot supply inference windows. Let

```math
\pi_t(L) = P_{\mathrm{LID}}(L \mid x_t).
```

Then

```math
\ell_t=\log\frac{\pi_t(E)+\epsilon}{\pi_t(M)+\epsilon},
\quad
\ell_\varnothing=\log\frac{\pi_\varnothing(E)+\epsilon}{\pi_\varnothing(M)+\epsilon},
\quad
\boxed{E_t=\left[\tanh\left(\frac{\ell_t-\ell_\varnothing}{2}\right)\right]_+},
\quad \epsilon=10^{-12}.
```

Here `\pi_\varnothing` is the same native-LID provider on an all-zero 30 s waveform under the frozen model, preprocessing and precision. `E_t=0` at the null/template odds, rather than leaving a 0.5 gate floor on Mandarin. This is **positive excess model evidence**, not a calibrated English probability or physical likelihood ratio. This local use of Whisper's segment-level language-ID mechanism is a **hypothesis to validate**, not an assumed fact. P0-R2 must check that the local-window construction is stable, reference-free, and distinguishes embedded-language regions from matrix-language regions. The 1.0 s window is frozen; invalid windows abstain with a reason, without outcome-based resizing.

### 3.4 P0-R2 hypothesis: baseline language conflict

Let the ordinary baseline next-token distribution be

```math
p_t^B(v) = P(v \mid x,y_{<t};c_B).
```

Define generically

```math
R_t = BaselineConflict(p_t^B;M,E), \qquad R_t \in [0,1].
```

For the first Mandarin-English feasibility test, use deterministic evaluator-independent token sets `V_M` and `V_E` defined from tokenizer/token text/script rules **before looking at R2 outcomes**. Exclude or explicitly classify mixed/ambiguous/special tokens. Compute

```math
P_M^B(t) = \sum_{v\in V_M} p_t^B(v),
\qquad
P_E^B(t) = \sum_{v\in V_E} p_t^B(v).
```

The raw matrix-over-embedded log odds are

```math
C_t =
\log\frac{P_M^B(t)+\epsilon}
          {P_E^B(t)+\epsilon}.
```

Use the positive **excess probability mass in the full next-token distribution**

```math
\boxed{
R_t =
\left[P_M^B(t)-P_E^B(t)\right]_+
}
```

Save `Q_t=P_M^B(t)+P_E^B(t)` and the conditional margin as diagnostics. Since `0≤R_t≤Q_t`, negligible language-bearing mass produces negligible conflict continuously; there is **no `Q≥0.5` hard cutoff**. `R_t` means **baseline excess matrix-script mass**, not error by itself. Freeze disjoint tokenizer classes, including Han-lead UTF-8 fragments in `V_M`; abstain at incomplete UTF-8 content prefixes.

The Mandarin-English vocabulary partition is a first feasibility instantiation, not the final multilingual definition. Before same-script CS-FLEURS transfer, freeze a language-pair-independent `BaselineConflict` provider (for example a conditioned full-distribution language-affinity construction) on development data only. Do not claim zero-shot multilingual generality from Han-vs-Latin partitioning.

### 3.5 Repair-need gate: when to repair

**Frozen after R2 (main method from P1 onward):**

```math
\boxed{
g_t = E_t R_t^B .
}
```

*Historical R2 candidate (rejected, now an ablation):* R2 added a same-prefix forced-English counterfactual conflict `R_t^{Ecf}` (same tokenizer classes) and `D_t=[R_t^B-R_t^{Ecf}]_+`, giving `g_t^{cf}=E_tD_t`. It was feasible but significantly worse than `E_tR_t^B` (paired ΔAUROC −0.11), so the frozen rule rejected it. `g^{cf}` survives only as a P2/P3 mechanism ablation.

Desired qualitative behavior:

| Case | `E_t`: local support for `E` | `R_t`: baseline matrix bias | desired `g_t` |
| --- | ---: | ---: | ---: |
| matrix-language token | low | high or any | low |
| already-correct embedded token | high | low | low |
| English-to-Han confusion | high | high | high |

English-to-English substitutions are out of this mechanism. Deletions have no emitted word; evaluate their next-token or EOS gap slot separately and conditionally.

Thus the selected gate asks whether **local acoustics support `E` while baseline decoding is `M`-like**. It is a ranking score, not a proof of repair. The detector formula is frozen; only the score-to-dose relation remains to be validated in P2.

### 3.6 Intervention

After P0-R2 validates the gate and P1 validates causal plumbing, use

```math
\tilde h_{\ell,t} = h^B_{\ell,t} + \alpha g_t d_{\ell,t},
```

followed by

```math
\boxed{
h'_{\ell,t}
=
\tilde h_{\ell,t}
\frac{\|h^B_{\ell,t}\|_2}
     {\|\tilde h_{\ell,t}\|_2+\epsilon}.
}
```

At exactly zero effective dose, bypass both addition and rescaling so baseline identity is exact.

### 3.7 P0-R2 scientific gates before P1

P0-R2 is a **gate-feasibility study**, not a steering experiment. It must evaluate the components separately before multiplying them.

- **R2-LS — LocalSupport validity.** On a predeclared development diagnostic panel with improved token/reference alignment, test whether `E_t` is dense/finite and separates embedded-language positions from matrix-language positions. Report full distributions, pairwise language posterior mass, local-window/alignment failures, and an audio-dependence control. If local native LID is unstable or nearly constant on short windows, block this instantiation rather than tuning many windows.
- **R2-BC — BaselineConflict validity.** Verify tokenizer-language partition coverage and ambiguity, finite full-distribution masses, and the `0≤R≤Q` invariant. Report `P_M/P_E/Q/R` by `EN-confusion`, `EN-deletion-slot`, `EN-same-language-sub`, `EN-correct`, and `ZH-correct`; no generic baseline-wrong English category or arbitrary `R≥0.5` boundary.
- **R2-ECF / R2-GATE — repair-need discrimination (historical R2 contract; outcome: `g^{cf}` not preferred, `E_tR_t^B` selected).** Evaluate both `g_t^{old}=E_tR_t^B` and `g_t^{cf}=E_t[R_t^B-R_t^{Ecf}]_+`, with the latter primary in R2, for `EN-confusion` versus `ZH-correct` and versus `EN-correct`. Report paired AUROC differences, deletion slots secondarily, and same-language substitutions as out-of-mechanism diagnostics while retaining all in overall ASR metrics. Reference labels are evaluator-only and never enter inference.
- **Alignment validity.** The previous coarse character-fraction position mapping produced noisy examples. R2 must use a documented, reproducible alignment with explicit unalignable cases; do not silently force labels onto generated positions. Alignment quality is part of the feasibility verdict.
- **No-reference / no-leakage.** Local audio spans, token sets, posterior calculations, and gate values must be computable without reference transcripts or oracle switch positions.
- **P1 entry condition.** P1 stays blocked until the final R2 gate contract is versioned and the above component/combined diagnostics are judged usable under predeclared criteria. A promising `hE-hM` direction alone is not sufficient.

### 3.8 Later cross-model/multilingual contract

The paper-level abstraction is

```math
d_{\ell,t} = norm(h^E_{\ell,t}-h^M_{\ell,t}),
\qquad
E_t = LocalSupport(E|x,t;M,E),
\qquad
R_t^B = BaselineConflict(p_t^B;M,E),
\qquad g_t=E_tR_t^B .
```

(The R2-rejected counterfactual `R_t^{Ecf}` / `g_t^{cf}` is an ablation, not part of the paper-level method.)

`LocalSupport` and `BaselineConflict` are provider interfaces, not permanently tied to Whisper local LID or Chinese-vs-Latin script. For a new model such as Qwen3-ASR, revalidate `cB/cM/cE`, common-prefix semantics, the hidden intervention site, local language evidence, and baseline-conflict provider on that model's development data before any locked evaluation.

## 4. Paper-first execution order

The critical path is now **P0/P0-R1 completed negative feasibility → P0-R2 revised gate feasibility → P1 causal working method → P2 compact development/freeze → P3 locked evaluation → P4 paper artifacts/audit**.

| Stage | Status / do now | Output / decision |
| --- | --- | --- |
| **P0 / P0-R1 — original feasibility** | **COMPLETE.** Preserve the K=1/K=3 candidate-support artifacts and the `cB==cM` result. Do not continue K sweeps. | Historical evidence: geometry collapse is non-identifiable; common-prefix plumbing passed; discrete candidate-continuation support is blocked. |
| **P0-R2 — revised repair-need feasibility** | **COMPLETE: `R2_CF_FEASIBLE_NOT_PREFERRED`, POST_R2_AUDIT PASS, selected `g_old=E·R_B` (job 54758).** The independent design audit, formula reconciliation, and repairability spec are complete. Commit and prepare the clean-state manifest, then evaluate the exact contract on the dense, existing 300-utterance D-dev-select panel. No steering intervention or alpha/layer tuning. | Freeze or reject the revised gate; compare old and counterfactual gates. Separate verdicts for local support, both conflicts, alignment, and combined repair-need discrimination. |
| **P1 — working causal method** | **COMPLETE: `P1_CAUSAL_ACCEPTANCE_PASS` (job 54781), POST_P1_AUDIT PASS → READY_FOR_P2_COMPACT_DEVELOPMENT.** After R2 passes, implement the smallest frozen-model path: per-prefix `hE-hM`, the R2-selected gate `g_selected=g_old=E_tR_t^B` (unchanged), norm-preserving edit at the existing engineering anchor, isolated branch/cache semantics, and standard reference-free decoding. | One integrated GPU acceptance covering alpha=0 identity, real eligible edit, cache/mask/beam behavior, direction/gate plumbing, and serialization. This is the first stage that tests an intervention. |
| **Pre-P2 — cached execution and contract freeze** | **COMPLETE: `CACHED_EQUIVALENCE: PASS` (spec v1.1, job 54802; attempt 1 job 54801 BLOCK by a CE6 instrument defect, preserved), `PASS_TO_P2_DEVELOPMENT` (`docs/inference_cf/PRE_P2_AUDIT.md`).** Implement a KV-cached steering path, pass a frozen cached-vs-replay equivalence acceptance, reconcile docs/provenance, and freeze the P2 contract before any P2 recognition outcome. | `CACHED_EQUIVALENCE` verdict; `PASS_TO_P2_DEVELOPMENT` or `BLOCK_BEFORE_P2`. |
| **P2 — compact development and freeze** | **COMPLETE: `P2_VALID_CONFIGURATION_SELECTED` → `READY_FOR_P3_LOCKED_EVALUATION` (spec v1.1; P2-A r1 jobs 54821/54822, P2-B 54824, P2-C 54825; attempt 1 jobs 54803/54804 `P2_BLOCKED_INVALID_EXPERIMENT`, preserved). Frozen: L16, α=2.0, φ_id, g=E·R_B, +d (`results/inference_cf/p2_frozen_configuration.json`, `docs/inference_cf/P2_REPORT.md`, `P2_AUDIT.md` PASS). Effect vs matched α=0 baseline only ΔPIER +2/2268 (CI95 [0, 0.0027]); B0_AUTO remains 8.2 PIER points better; P2-B controls (energy-matched g=1 +5, −d +11, g=R_B +10) do not support detector selectivity or the +d direction. `OPEN DECISION`: human review before P3.** On D-dev-select, with the frozen detector `g=E·R_B`: P2-A efficacy screen over layers {16,24} × α∈{0.5,1.0,2.0} (φ(g)=g); P2-B mechanism controls (g=1, g=E, g=R_B, g=E·R_B, g_cf, −d, energy-matched constant); P2-C one dose map φ(g)=√g vs g, only if P2-A finds a valid configuration. Constraints and tie rule are frozen in `docs/inference_cf/P2_COMPACT_DEVELOPMENT_SPEC.md` before outcomes. No detector search. | Freeze deployable single-layer configuration, comparator settings, damage constraints, selection rule, hashes, and attempted settings. |
| **P2-R — mechanism diagnosis** | **COMPLETE: `P2_R_MECHANISM_STILL_AMBIGUOUS`, `P2_R_AUDIT: PASS` (spec `docs/inference_cf/P2_R_MECHANISM_DIAGNOSIS_SPEC.md`, job 54874; attempts 54855/54871 preserved). No repair hypothesis; P3 HELD.** Diagnose (not tune) why P2 showed no selective repair: D1 +d/−d at matched realized energy on frozen evaluator positions, D3 one frozen random direction, D2 current vs energy-packet oracle placement. Oracle/reference information stays evaluator-only. | One primary diagnosis (or ambiguity) under frozen rules; at most one separately versioned repair hypothesis; no layer/α/gate/localizer/direction search. |
| **P2-RJ — Jacobian H1-vs-H4 disambiguation** | **COMPLETE: `P2_RJ_STILL_AMBIGUOUS`, `P2_RJ_AUDIT: PASS` (spec `docs/inference_cf/P2_RJ_JACOBIAN_DIAGNOSIS_SPEC.md` v1+v1.1, job 54998). Leverage UNRESOLVED (Q50 0.61 [0.44, 0.76]); alignment MISALIGNED (κ₊ −0.011 [−0.022, −0.001]); first-order validated (r 0.92). No repair; P3 HELD.** At the 180 frozen P2-R states: gradient of the evaluator-only reference margin at the L16 site, tangent norm × e* vs the lexical gap (H4), and alignment of +d/−d/random realized edits with it (H1). Gradient never applied as steering; fp32 precision replicate. | One frozen diagnosis (DIRECTION / SITE / DIRECTION_AND_SITE / AMBIGUOUS); at most one diagnosis-derived repair hypothesis, never validated without an authorized fresh role. |
| **P2-RJ-E — final leverage expansion (diagnostic program end)** | **COMPLETE: `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED` (terminal), `P2_RJ_E_AUDIT: PASS` (spec `docs/inference_cf/P2_RJ_E_LEVERAGE_EXPANSION_SPEC.md`, job 55002). Q50 0.539 [0.406, 0.665] on 227 positions (60 original bitwise-reproduced + 167 new); UNRESOLVED in bf16 and fp32. Diagnostic program ENDED.** Identical P2-RJ protocol; only the sample expanded to every P2-R-eligible EN-confusion position of D-dev-select. | One terminal state; then an independent actuator-redesign pass (`P2_RJ_E_ACTUATOR_REDESIGN_HANDOFF.md`). No further P2-R-family diagnostics. |
| **Actuator redesign (next; design only)** | **NOT STARTED — awaiting one independent Codex/Astra design pass.** Direction-first replacement for norm(hE−hB) at L16 with a predeclared site-failure criterion; detector may stay fixed for the first repair experiment; evaluator/oracle signals forbidden at inference; validation role requires human authorization. | One frozen, versioned actuator spec before any implementation. |
| **P3 — locked evaluation** | Evaluate frozen systems according to Section 9's corpus-by-system matrix, including gate ablations and monolingual guard. | Core recognition, component evidence, preservation, uncertainty and cost results. No retuning on test/transfer splits. |
| **P4 — paper artifacts and audit** | Regenerate tables/figures from saved records; expand multilingual/two-architecture claims only after the required transfer/replication evidence. | Auditable paper package; negative transfer/model results narrow claims rather than triggering test-time redesign. |

**Minimum development comparisons after R2/P1:** (1) standard ASR (B0 forced-ZH and B0_AUTO); (2) embedded-language-forced decoding without steering; (3) dynamic `hE-hM` with constant gate/dose; (4) the R2-selected gate `g=ER_B` at one layer (main method); (5) the R2-rejected counterfactual gate `g_cf=E[R_B-R_Ecf]_+` only as a reported ablation, `LocalSupport`-only (`g=E`), and baseline-conflict-only (`g=R_B`) at matched/appropriately tuned dose; (6) reversed direction at matched dose; (7) the all-layer comparator. Do not resurrect the retired `rho/c` collapse ablation. Additional random vectors, broad window sweeps, extensive gate-temperature grids, 32 individual behavioral layer scans, retraining, and several speech-LLMs are outside the critical path.

### Layer policy: single-layer primary and one all-layer comparator

The primary deployment candidate is a **single layer chosen on development data**. This makes the per-token intervention local and its damage/cost easier to attribute. The all-layer condition remains a scientific comparator. Collect unsteered branch states and gate signals consistently before applying a frozen all-layer edit schedule; otherwise deeper counterfactual states after earlier edits do not have the same semantics.

Use a separately **development-tuned all-layer strength** and compare single versus all layers at comparable realized per-token intervention energy, also reporting each method's best development-valid configuration. Applying the full single-layer alpha at every layer is an uncontrolled dose increase. Freeze both recipes before test and evaluate both on the same locked primary corpus. Include PIER, MER, English WER, Mandarin CER, retention, correction/corruption, gate coverage and runtime. Geometry-only 32-layer diagnostics are optional if cheap; never use test results to select a layer.

**Resource rule:** use `sbatch` for GPU work; inspect existing jobs and cap this task at two pending/running GPU jobs. Prefer the validated MIG profile when measured footprint fits. R2 should use one predeclared physical feasibility run after its design audit, with reruns only for concrete implementation/infrastructure invalidity. Batch compatible model calls, cache only scientifically identical states/posteriors, use atomic shards, and resume only missing validated rows. CPU aggregation needs no GPU.

### Main-page outputs to prepare from P2–P4

Section 9.6 remains the final layout target: three main tables (recognition, method evidence/cost, generalization) and two figures (algorithm, correction-damage/gate evidence). The algorithm figure must show the **validated** local support × counterfactual conflict reduction gate rather than the retired `rho/q_cross` design. Do not produce paper figures from feasibility rows or select favorable test settings. Save source tables for every plot.

### Conditional appendix expansion after P3

These experiments follow the frozen Whisper result and must not influence P2 configuration selection. Report aggregate conclusions briefly in the main text only if the paper claims multilingual breadth or cross-architecture applicability.

| Priority | Experiment | Protocol and minimum output | Gate / claim |
| --- | --- | --- | --- |
| **A1 — CS-FLEURS language breadth** | Use a documented official X-English test scope. Apply the frozen *paper-level* `LocalSupport × BaselineConflict` algorithm without per-language outcome tuning. Before this locked transfer, the provider used for same-script pairs must already be language-pair-independent; the Mandarin Han-vs-Latin token partition alone cannot support a general multilingual claim. Report per-language N, embedded-target error, matrix-language error, paired differences, gate activation and cost; keep same/different-script groups visible. | Tests cross-language application of the algorithm. Any unsupported native LID/prompt/provider pair is reported as ineligible/negative feasibility, not silently replaced after test outcomes. |
| **A2 — ViMedCSS domain transfer** | Use the official Vietnamese-English test split; set `M=Vietnamese`, `E=English`, and apply the frozen provider/method. Report term correction, Vietnamese preservation, gate behavior and latency. | Tests language + medical-domain transfer together; isolate neither effect causally. |
| **A3 — Qwen3-ASR architecture replication** | Fix the model/checkpoint. First re-run architecture feasibility: ordinary vs forced M/E conditions, same-content-prefix semantics, comparable hidden site, model-native/local language evidence, and a valid BaselineConflict provider. Select Qwen layer/scale only on Qwen development data under a separate protocol, then freeze. | Cross-architecture replication; do not assume Whisper's `cB==cM`, layer 24, local-LID behavior, alpha, or tokenizer-language mapping transfers. |

**Efficient order:** finish P0-R2 → P1–P3 Whisper core → prepare outcome-blind transfer manifests/provider compatibility checks on CPU → A1/A2 → A3. Do not run transfer suites while deciding the primary Whisper gate.

Runtime projections must be regenerated after P1 because local-LID/window extraction plus counterfactual branches change the cost model. Reuse only scientifically identical baseline encodes/decodes/posteriors.

## 5. Data roles, metrics, and claims

| Role | Data | Allowed work |
| --- | --- | --- |
| Engineering/selection | Existing CS development split(s), preserving construction/selection/confirmation distinctions and the complete exposure ledger | P0-R2 gate feasibility, alignment repair, debugging, and P2 parameter selection. Record every exposure; do not call reused examples untouched. |
| Additional development | ASCEND official validation, only after local manifest/CS eligibility are verified | Optional development confirmation/selection if declared before outcomes. |
| Locked confirmation | CS test and official ASCEND test, if access/role is documented and approved | Inference/evaluation only after configuration freeze. |
| Transfer | SEAME-man/sge, CS-FLEURS, ViMedCSS | Apply the appropriate frozen algorithm/provider according to the predeclared transfer protocol; no outcome-driven per-language tuning. |

Evaluate ordinary decoding and method on identical eligible IDs. Report micro MER, PIER where canonically defined, embedded English WER/target metric, matrix-language error, corrections/corruptions, outside-target harm, transcript edits, matrix/embedded retention, gate coverage with explicit denominator, edit norms, extra model/LID passes, latency, peak VRAM and GPU-hours. Define `Delta = baseline - method` so positive means improvement. Final locked tests use paired bootstrap intervals with the independent sampling unit specified in Section 9.4.

For gate diagnostics, keep reference labels and alignments strictly evaluator-side. Report `E_t`, `R_t^B`, the selected `g_t=E_tR_t^B` (and, where the ablation is run, `R_t^{Ecf}`, `D_t`, `g_t^{cf}`), raw/null language posterior/log-odds components, both `Q_t` values, ambiguous-token mass, local-window/alignment status, and offline category labels separately. Do not call either gate a calibrated probability of error unless calibration is explicitly established. Report B0_AUTO as a mandatory matched-ID unsteered comparator to the known-M B0 and proposed method.

Keep baseline and steered decode settings identical within each model (model revision, beam/temperature, language/task prefix, token limit, cache mode and timestamp policy). Greedy remains allowed for plumbing/development when labeled; final comparison uses the frozen standard setup. For Whisper, document that ordinary Mandarin-dominant decoding may be computationally identical to explicit Mandarin conditioning.

## 6. Suggested interface and outputs

Use repository conventions, but the revised method should expose separate interfaces for:

- `ConditioningPolicy`: `cB/cM/cE` prompts and same-content-prefix mapping.
- `StateProvider`: baseline/M/E hidden states and cache identity.
- `Localizer` or `AudioSpanProvider`: deterministic current-token acoustic window with alignment provenance.
- `LocalSupport`: positive null-relative embedded-language evidence `E_t`.
- `BaselineConflict`: full-distribution excess matrix-script mass under B0 (and, for the ablation only, same-prefix Ecf), including `Q_t` and tokenizer coverage diagnostics.
- `Gate`: frozen main detector `g_t=E_tR_t^B` with explicit fallback behavior; `g_t^{cf}=E_t[R_t^B-R_t^{Ecf}]_+` is the R2-rejected ablation.
- `DoseMap`: `s_t=α·φ(g_t)` with `φ(g)=g` (reference) and, in P2 only, the single predeclared alternative `φ(g)=√g`.
- `SteeringHook` / `Decoder`: only from P1 onward.
- `Evaluation` / `RunManifest`: immutable population/config/provenance accounting.

For the primary Whisper R2 implementation, cache keys must cover model/tokenizer revision, audio identity, local span/window rule, condition, exact prompt tokens, content prefix, logical position, layer/site when applicable, beam lineage, LID/provider version and decoder config. Never reuse a local LID posterior or hidden state across a semantically different prefix/span/condition.

P0-R2 row artifacts must preserve enough scalars to recompute without GPU:
`pi_M`, `pi_E`, null posteriors, raw/null log odds, signed evidence margin, `E_t`, B0/Ecf `P_M/P_E/Q/R`, ambiguous/unclassified mass, diagnostic conditional margins, `D_t`, `g_t^{old}`, `g_t^{cf}`, structural fallback reason, exact maximum-mass window and point-window diagnostic, baseline token distribution summaries, and evaluator-only category/alignment fields. A bounded diagnostic sample may retain full distributions if needed; do not permanently cache full-vocabulary logits for the entire corpus unless justified.

From P1 onward, one JSONL record per utterance/system should additionally contain baseline/method hypotheses, direction/gate summaries, realized edit norms, metric counts, timing and failures. Keep one manifest with model/dataset/config/source hashes and freeze timestamp. Never overwrite historical P0/P0-R1 result directories.

## 7. Tests with actual failure modes

### P0-R2 focused CPU/unit tests

Protect the specific new risks:

- exact `cB/cM/cE` serialization and preservation of the historical `cB==cM` finding for the primary Whisper config;
- deterministic token/audio local-span extraction inputs/outputs and boundary clipping;
- no reference transcript/oracle span passed to `Localizer` or `LocalSupport`;
- native LID language-index/token mapping, null-template subtraction, and exact zero at null odds;
- `E_t` finite/bounded behavior for zero/tiny pairwise mass and nonpositive evidence;
- deterministic construction/versioning of `V_M`, `V_E`, ambiguous/special sets;
- vocabulary coverage accounting and no double classification unless explicitly allowed;
- `P_M^B/P_E^B` aggregation from a controlled probability vector;
- full-distribution mass identity `R_t=max(0,P_M-P_E)`, `0≤R_t≤Q_t`, including low-`Q_t` cases;
- `R_t=0` when embedded mass is at least matrix mass and `0<=R_t<=1`;
- `D_t=[R_t^B-R_t^{Ecf}]_+`, `g_t^{cf}=E_tD_t`, old gate reproduction, exact zero and fallback rules;
- improved evaluator-only token/reference alignment with explicit unalignable outcomes;
- cache keys include audio/span/provider/prefix/condition/beam/model identity;
- row schema, atomic resume/dedup, and manifest-hash mismatch refusal.

### P0-R2 integrated GPU feasibility

After the independent design audit and committed pre-run spec, use one small development run to verify:

- local audio windows correspond sensibly to current generated positions and are reproducible;
- local LID returns nontrivial finite M/E posterior mass;
- `E_t`, both conflict scores, repairability, and both gates are dense enough for analysis rather than collision-dominated;
- component/combined score distributions can be recomputed from saved rows;
- reference labels never enter the inference path;
- measured cost/VRAM/model-call counts are captured.

Do not steer activations in R2.

### P1 causal acceptance

Only after R2 passes: alpha=0 exact identity; finite sample-varying direction; one eligible nonzero edit at the intended state/site; no prefix corruption; norm preservation; correct cache advance/beam lineage; serialization; measured memory/cost. Add focused regression tests only for concrete failures; do not schedule broad smoke suites by habit.

## 8. Standing instructions for the next coding assistant

At the start of every pass:

1. Read this file, all applicable `AGENTS.md` / `CLAUDE.md`, `docs/inference_cf/P0_R2_DESIGN_AUDIT.md`, `docs/inference_cf/P0_R2_FORMULA_RECONCILIATION.md`, `docs/current/{METHOD_CONTRACT,CODE_MAP,DATA_EXPOSURE,STATUS}.md`, `FEASIBILITY.md`, `P0_INDEPENDENT_AUDIT.md`, and the P0-R1 report/spec. Inspect Git status/HEAD and recent relevant commits. The current user-approved request and repository rules take precedence.
2. Preserve historical P0/P0-R1 artifacts. Never rewrite the negative feasibility evidence to match the new hypothesis.
3. Work one stage at a time with small reviewable commits. Any change to `LocalSupport`, `BaselineConflict`, local-span semantics, token classification, or gate composition is a scientific-contract change and must be versioned before looking at corresponding GPU outcomes.
4. Respect the data-role firewall and strict reference-free inference API. References may be used only after inference for diagnostic labels/metrics.
5. Return: stage/status; files changed; scientific assumptions confirmed/rejected; commands/tests/jobs; evidence/limitations; Git commit; exact next stage or blocker.

**Next concrete task:** Run ONE independent Codex/Astra actuator-design session using `docs/inference_cf/P2_RJ_E_ACTUATOR_REDESIGN_HANDOFF.md`. Do not implement a replacement actuator before that design pass. P3 stays HELD. The diagnostic program is ENDED (P2-RJ-E `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`): d=norm(hE−hB) is not a useful lexical-repair direction at L16 (captures ≈ −1% of the available first-order leverage; −d not validated); L16 leverage at the P2 budget is unresolved (Q50 0.539 [0.406, 0.665] after exhausting the eligible D-dev-select population). No fresh repair-validation role is authorized (`router-calib` only with human authorization). *(Superseded pointer, P2-RJ close:)* `OPEN DECISION` (human). P3 stays HELD. P2-RJ (`docs/inference_cf/P2_RJ_REPORT.md`) ended `P2_RJ_STILL_AMBIGUOUS` under its frozen rules, but narrowed the ambiguity: the direction d=norm(hE−hB) is near-orthogonal to the useful lexical (largely script-readout) direction at L16 (captures ~−1% of available first-order leverage; H1 holds whichever leverage class applies), and the remaining open question is only whether L16 leverage at e* is sufficient (median 58% of the gap closable; Q50 straddles 0.5). Single most information-efficient follow-up (needs authorization): the identical P2-RJ protocol on a predeclared expansion to all remaining structurally valid EN-confusion positions of the D-dev-select panel, deciding only Q50 (→ DIRECTION_ISSUE vs DIRECTION_AND_SITE_ISSUE, which select different one-repair policies). Alternatives for the human: proceed directly to a single direction-construction hypothesis (H1 is established), redesign the representation/site, or narrow the project; P3 locked evaluation of the P2 configuration is not recommended. *(Superseded pointer, P2-R close:)* `OPEN DECISION` (human). P3 stays HELD. P2-R (`docs/inference_cf/P2_R_REPORT.md`) ended `P2_R_MECHANISM_STILL_AMBIGUOUS`: the L16 intervention is ~inert at the lexical decision at P2 energy, and +d is the weakest direction tested. The single most informative next diagnostic (H1 direction vs H4 site/sensitivity) is the evaluator-only reference-margin Jacobian at the same 180 frozen positions/states (tangent norm × e* vs the 10.6-nat gap; alignment of d vs random with the gradient; never applied as a steering direction). Alternatives for the human: authorize that follow-up, redesign the representation/site, or narrow the project; P3 locked evaluation of the P2 configuration is not recommended on current evidence.

## 9. Paper-completeness contract

### 9.1 Declared scope and claims

The core paper tests **training-free, reference-free-at-inference code-switching repair** using (i) utterance/prefix-conditioned counterfactual directions and (ii) a selective repair-need gate that combines local embedded-language acoustic support with baseline language contradiction. The main causal claim is conditional on both gate feasibility and intervention results; P0/P0-R1 alone do not prove the direction repairs ASR.

For Whisper Mandarin-English, the earlier `rho/c` matrix-collapse factor is retired because baseline and explicit matrix conditioning are identical. The earlier branch-generated continuation support is retained only as negative feasibility evidence. The revised method must not be described as if these components survived.

The first R2 `BaselineConflict` implementation may use Mandarin/English tokenizer-language groups because the scripts differ. This does **not** justify a general multilingual claim. A same-script-capable provider must be frozen before CS-FLEURS breadth evaluation. Similarly, local Whisper LID is a model-specific instantiation of `LocalSupport`, not the paper-level definition.

The expanded paper may test CS-FLEURS, ViMedCSS, SEAME and Qwen3-ASR only after the relevant provider/model feasibility contracts are frozen. The matrix/embedded language pair is an input assumption unless a separate detection mechanism is explicitly added. Reference-derived token language, switch positions, words and future tokens never become inference inputs.

### 9.2 Freeze this experiment matrix before P3

Do not run every system on every corpus. Use the primary CS test for mechanistic controls, both approved core tests for competitive baselines, and transfer suites for the frozen deployable method. All rows below must be frozen independent of whether outcomes are favorable.

| ID | System / evidence | Required population | Question answered |
| --- | --- | --- | --- |
| B0 | Ordinary ASR under the declared baseline condition | Every evaluated corpus/model | Is there a gain over actual deployment baseline? |
| B1 | Embedded-language-forced decoding without edits; include matrix-forced separately only when distinct from baseline | Both core tests; Qwen primary test | Is steering better than simply forcing a language? |
| B2 | Valid bilingual/context prompt without transcript/test vocabulary leakage | Both core tests; Qwen primary test | Is a simple prompt sufficient? |
| B3 | Reference-free output-distribution mixture using the same M/E branches | Both core tests; Qwen primary test | Do hidden-state edits add value beyond branch predictions? |
| M | Dynamic `hE-hM` direction + frozen detector `g=E·R_B` with the P2-selected dose map `α·φ(g)`, selected single layer | Every approved corpus/model with a valid provider | Main deployable method (R2 and P1 passed; P2 selects layer/α/φ) |
| C1 | Dynamic direction, constant gate/dose; separately tuned constant dose and a development energy-matched dose when needed | Primary CS test | Is localization useful beyond reducing total perturbation? |
| C2 | `LocalSupport` only (`g=E`) | Primary CS test | Does baseline contradiction suppress unnecessary edits on already-correct embedded positions? |
| C3 | `BaselineConflict` only (`g=R`) | Primary CS test | Does local acoustic evidence suppress matrix-language over-steering? |
| C3b | R2-rejected counterfactual gate `g_cf=E[R_B-R_Ecf]_+` | Primary CS test | Does the counterfactual restriction add intervention-level value despite worse R2 ranking? |
| C4 | Reversed direction with same gate/site/dose | Primary CS test | Does direction sign matter? |
| C5 | Simultaneous all-layer steering with its own frozen development strength | Primary CS test | Is one site sufficient relative to distributed intervention? |
| C6 | One fixed direction per current utterance from an unsteered draft, same final gate | Primary CS test | Does token-varying direction add value beyond utterance personalization? |
| C7 | Corpus-fixed mean conditioned displacement built from declared development donors, same final gate | Primary CS test | How does a conventional construction-data fixed direction compare? |
| D1 | `E_t`, `R_t^B`, selected `g=E·R_B` (plus Ecf/`D_t`/`g_cf` ablation diagnostics), direction diagnostics + audio/localization controls on generated baseline prefixes | Predeclared diagnostic panel | Does the gate target the claimed repair cases and depend on speech? |
| D2 | Matrix-language-only speech: B0 and M | Fixed held-out monolingual panel | Does the method introduce embedded-language insertions or damage ordinary speech? |

**B3 default:** for the same audio/content prefix obtain `pM` and `pE` over the common vocabulary and decode from a predeclared mixture family. On development compare a small fixed lambda grid and, if well-defined, a gate-linked mixture such as `lambda=g_t`; freeze one recipe before test. Do not reuse independently generated branch histories as the current mixture prefix.

**C6/C7:** pool raw conditioned displacements at the selected site excluding prompt/padding/EOS. C6 uses only the current utterance and generated draft, no gold text. C7 uses an outcome-independent development donor population and fixes the mean before dose selection. Keep the same gate definition across M/C6/C7 and comparable development budgets.

**Dose control:** derive a constant-gate scale from development gate RMS as a starting energy match and record realized edits after NormPreserve. For all-layer comparison use summed squared realized relative edits across edited sites per logical position. Freeze matching rules/scales on development; do not optimize them on test.

Freeze all comparator grids, tie rules, damage tolerances, provider versions and fallback semantics before test. If runtime requires a smaller ablation population, choose deterministic IDs before inference and report it as a subset, not full-test evidence. Never claim SOTA from this controlled matrix unless a separate fair benchmark justifies it.

### 9.3 Diagnostics to collect during planned runs

1. **Gate discrimination:** on the frozen baseline-prefix panel save `E_t`, both conflicts, `D_t`, both gates, their raw posterior/mass components, local-span/alignment identity, prefix/token/beam identity, and evaluator-only alignment to (a) `EN-confusion`, (b) `EN-deletion-slot`, (c) `EN-same-language-sub`, (d) `EN-correct`, and (e) `ZH-correct`. Report distributions, PR/AUROC and paired old/new differences when labels are reliable, Mandarin/matrix activation, and unalignable cases. These are predictive associations; end-to-end ablations provide causal intervention evidence.
2. **LocalSupport validity/audio dependence:** record pairwise native-LID posteriors, local window boundaries and failures. Use one predeclared audio control (for example mismatched or degraded local audio) only as a diagnostic to test whether the support signal responds to speech. A change under artificial audio does not by itself prove a calibrated acoustic likelihood.
3. **BaselineConflict validity:** report both branches' `P_M/P_E/Q/R`, ambiguous/unclassified mass, raw log odds, tokenizer-language coverage and failure cases. For same-script/generalized providers, save the corresponding language-affinity components.
4. **Sensitivity:** export already-run development layer/dose results and R2 provider/window decisions including failures. The old point-argmax window is a predeclared localization diagnostic, not a gate-outcome-selected fallback; do not run a favorable-window search.
5. **Error behavior:** derive corrections, corruptions, persistent errors, insertions/deletions, translation/transliteration/script conversion and repetition/truncation flags from saved transcripts. Manually audit a small predeclared changed-output sample when automatic categories are ambiguous. Do not equate more Latin output with correct English recognition.
6. **Monolingual guard:** save matrix-language error, embedded-language insertion/false gate activation and runtime for D2. Matrix segments inside CS speech do not replace utterances containing no embedded language.

### 9.4 Evaluation validity and uncertainty

- Verify all historical exposure before labeling a split untouched. A corpus called `test` may already have informed A6/A7 or prior design. Preserve the exposure ledger; call it held out from the current tuning only when that is true. If independent confirmation is claimed, include a genuinely unconsulted evaluation population and freeze choices before its outcomes are viewed.
- Freeze raw IDs, source revisions, speaker/dialogue/session IDs where available, preprocessing and exact eligible populations. Inference eligibility must not depend on successful baseline English recognition, an oracle mask, alignment success or low baseline error. A valid input with an undefined direction gets a documented runtime fallback and stays in end-to-end evaluation. Invalid audio and genuinely undefined target metrics are counted explicitly. Empty/failed hypotheses must not be silently dropped; freeze their scoring/fallback rule and report failure rate.
- Keep references and language/term annotations in evaluator-only inputs. In same-script pairs, use validated target annotation/lexical alignment for English metrics rather than a Latin-character rule. Use the dataset's declared word/character segmentation and punctuation/case/number normalization; PIER is only valid when canonical target units exist. State denominators and no-target behavior (not an invented zero). Distinguish whole-utterance, embedded and matrix errors. A pair with changed scoring units is not directly comparable in absolute error to another pair.
- Default primary endpoint: corpus PIER improvement where canonical POIs exist, with MER and matrix-language harm as co-reported outcomes; elsewhere predeclare the appropriate embedded-target metric. Before dev selection, fill numerical matrix-damage tolerance, minimum retention, primary metric, corpus weighting, selection/tie rule and no-valid-setting fallback in the contract. These fields must not stay implicit until favorable results appear.
- Save numerator/denominator counts and recompute corpus micro metrics inside each paired bootstrap resample. Use at least 2,000 fixed-seed replicates. Predeclare the independent sampling unit: dialogue/session for conversational data or speaker/source-sentence clusters where appropriate; utterance bootstrap is appropriate only when its independence assumption is defensible. If cluster metadata is unavailable, disclose the limitation. Bootstrap intervals quantify sampling uncertainty, not uncertainty over the chosen hyperparameter search.
- Define the primary significance comparison before test. For many language/comparator claims, label intervals pointwise and use a declared multiplicity correction (for example Holm on paired-test p-values) when asserting a family of significant wins. Report full outcomes, win/tie/loss counts and macro language effects; do not treat 12 languages or 32 layers as interchangeable utterance replicates. Preserve source-sentence grouping across languages if constructing an aggregate interval from parallel material.
- Deterministic inference needs no habitual three-seed reruns. If a required control is randomized, predeclare its seeds and report variability; a random control is otherwise optional because the sign and fixed-direction controls are already specified.

### 9.5 Efficiency and result capture: avoid missing-data reruns

Use identical hardware allocation, precision, batch policy and output limits for within-model runtime comparisons; record exact MIG profile. Separate model loading/warm-up from steady-state time and diagnostic logging from ordinary baseline cost. Report end-to-end latency (mean/p50/p95), real-time factor, processed audio duration, output-token count, peak VRAM, encoder/decoder evaluations, local-LID evaluations/window extraction cost, conditioned branch evaluations and total GPU-hours.

Before any long run, verify each shard saves raw output token IDs/text, termination/failure status, reference join ID, evaluator alignment/count records, `E/R/g` summaries, local-window/provider provenance, dose/norm summaries, runtime, config hashes and beam/prefix provenance. Save enough posterior/mass components to recompute the gate from CPU artifacts. Full hidden tensors or full-vocabulary logits are unnecessary except a bounded diagnostic sample. Bootstrap, tables, figures and error breakdowns should be reproducible from saved compact outputs alone.

Profile on a fixed representative panel during P1/P2 and accumulate timing during evaluation; do not add a new full-corpus speed run merely for presentation. Different model defaults are allowed across architectures if justified, but each model's baseline and method must use the same frozen decoding regime.

### 9.6 Final main-page and appendix package

**Main pages (three tables and two figures; combine tables if page limits require):**

- **Table M1 — recognition:** B0–B3 and M on the two core held-out corpora; compact overall/embedded/matrix error columns and intervals for principal deltas.
- **Table M2 — method evidence and cost:** M, constant gate, `g=E`, `g=R`, fixed-direction controls and all-layer result; key error/damage/cost columns. Reverse-sign and detailed energy accounting may move to appendix.
- **Table M3 — generalization summary:** separate SEAME rows, CS-FLEURS language-macro summary with language count/N, ViMedCSS, and Qwen block. Keep metric units distinct and expose provider feasibility/failures.
- **Figure M1 — actual implemented algorithm:** local acoustic support `E` + baseline matrix-script conflict `R_B` → `g = E·R_B`; same-prefix hidden-state displacement `hE − hB` → direction `d`; `g` + `d` → localized, norm-preserving intervention at the selected site. Do not show retired `rho/c`, continuation `q_cross`, or the R2-rejected Ecf conflict-reduction gate as the final method (those belong to negative/ablation evidence).
- **Figure M2 — two panels:** (a) development correction-damage curve with frozen test operating points; (b) held-out `E/R/g` distributions for wrong embedded, correct embedded and matrix positions. If alignment is not reliable enough, replace panel (b) with an auditable transition summary and state the limitation.

**Appendix:**

- **A. Reproducibility/data:** formulas, local-span rule, native-LID/provider semantics, token classification/generalized affinity rules, prompts, state/cache/beam semantics, configs, data roles/exposure, hardware and all attempted development settings.
- **B. Full core results:** denominators/counts, errors, corrections/corruptions/outside harm/retention, failures, paired intervals and C1–C7 controls.
- **C. Sensitivity/provider evidence:** compact dev layer/dose results, all-layer energy/runtime, R2 provider/window evidence and failures; no broad hidden favorable search.
- **D. Multilingual/domain:** all frozen CS-FLEURS rows, script grouping, ViMedCSS and separate SEAME splits; no silent negative-language removal. Explicitly distinguish the primary Mandarin script-based conflict implementation from any generalized same-script provider.
- **E. Qwen replication:** model/site/prompt adaptation, `cB/cM/cE` feasibility, LocalSupport/BaselineConflict adaptation, dev selection, baseline/prompt/mixture/method results and failures.
- **F. Gate, preservation and failure analysis:** D1/D2, audio/localization control, gate components, error taxonomy and representative good/bad/null cases.
- **G. Cost:** latency/RTF/memory/forward/local-LID counts, construction-data access for fixed-vector comparators, total search/evaluation compute.

### 9.7 Completion checklist and stopping rule

- [x] Historical P0/P0-R1 conclusions preserved: `cB==cM` geometry retired for Whisper; continuation evidence recorded as blocked; common-prefix plumbing evidence retained.
- [x] P0-R2 repairability design, maximum-mass localizer, LS-B, BC-B, same-prefix Ecf and both gate formulas versioned before GPU outcomes.
- [x] R2-LS, R2-BC, alignment and combined gate feasibility resolved with no reference leakage (job 54758: `R2_CF_FEASIBLE_NOT_PREFERRED`, selected `g=E·R_B`, POST_R2_AUDIT PASS); P1 began only after that verdict.
- [x] P1 causal plumbing passes exact zero-dose identity and a real eligible edit (job 54781, `P1_CAUSAL_ACCEPTANCE_PASS`, POST_P1_AUDIT PASS); direction usefulness is not claimed from P0/P1.
- [x] Pre-P2 cached-execution equivalence (job 54802, PASS) and P2 contract (`P2_COMPACT_DEVELOPMENT_SPEC.md`) frozen before any P2 recognition outcome.
- [x] P2-RJ-E final leverage expansion complete (job 55002, audit PASS): `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`; diagnostic program ended; redesign handoff written; P3 held.
- [ ] Independent actuator-design pass (Codex/Astra) from `docs/inference_cf/P2_RJ_E_ACTUATOR_REDESIGN_HANDOFF.md`.
- [x] P2-RJ Jacobian H1-vs-H4 diagnosis complete (job 54998, audit PASS): `P2_RJ_STILL_AMBIGUOUS` (alignment MISALIGNED, leverage UNRESOLVED); no repair; P3 held pending a human decision.
- [x] P2-R mechanism diagnosis complete (job 54874, audit PASS): `P2_R_MECHANISM_STILL_AMBIGUOUS`; no repair; P3 held pending a human decision.
- [x] P2 compact development complete under frozen rules (v1.1 matched baseline after the recorded attempt-1 invalidity): selected L16 α=2.0 φ_id g=E·R_B (+d); P2-B/P2-C reported; independent audit PASS. Mechanism controls negative (`OPEN DECISION` before P3).
- [ ] Scope, metrics, damage tolerances, baseline/ablation matrix, provider versions and search budgets frozen before test.
- [ ] B0–B3 and M evaluated on approved core corpora, or missing populations and narrower claims explicitly recorded.
- [ ] C1–C7 evaluated where mathematically applicable; retired/non-applicable components have documented reasons.
- [ ] D1 gate/provider diagnostics and D2 monolingual guard complete; latency/memory include local support, conflict computation and all conditioned branches.
- [ ] Expanded A1/A2/A3 and SEAME evidence complete for any corresponding breadth/model claim, or paper scope explicitly narrowed.
- [ ] All expected IDs accounted for; failures/missing/duplicates resolved; gate/alignment denominators and bootstrap populations validated.
- [ ] Main/appendix tables and figures regenerate from saved outputs; claims link to actual evidence.
- [ ] Config/data/code/provider hashes, manifests and report committed; future sessions can reproduce without re-decoding completed rows.

Once these pass, stop routine experiments and write. Additional GPU work is justified only by a concrete bug, unresolved scientific contradiction, new desired claim, or inadequate statistical precision. Recompute resource projections from measured R2/P1 throughput rather than earlier candidate-continuation estimates.

## P2-DIR — frozen next implementation stage (2026-10-06)

The independent actuator-design pass is complete. Authoritative additive stage spec:
`docs/inference_cf/P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md`; machine config:
`configs/inference_cf/p2_dir_direction_identification.json`; actual code map/implementation
sequence: `docs/inference_cf/P2_DIR_CODEX_DESIGN.md`. Exactly D0 OLD, D1 UNIQUE, D2 READOUT;
L16/current gate/localizer/alpha2 remain fixed. Exp1 matched-energy pulses selects at most
one new direction; only audited pass permits Exp2 gate coupling then Exp3 development decoding.
No scientific GPU run in the design session, no new actuator implemented. Terminal P2-RJ-E
diagnosis remains accepted, diagnostic program ended, P3 held. This stage does not revise
DG-03R core-paper authority. Claude must follow frozen tests/thresholds/firewall/audit rules
and commit/push reviewed implementation before scientific outcomes.

## P2-DIR — executed (2026-10-06): `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`

| Stage | Status |
|---|---|
| P2-DIR implementation + pre-run audit | DONE (`1802441`, `00dd8b2`, `PASS_TO_P2_DIR_EXP1`) |
| P2-DIR Exp-1 (D0/D1/D2 direction-only, matched e*) | DONE, audited PASS: no candidate qualifies; D1 safe but +0.23 nat < 0.5; D2 +4.5 nat with ZH damage (causal power with damage) |
| P2-DIR Exp-2 (gate coupling) | NOT RUN (stop rule) |
| P2-DIR Exp-3 (free decoding) | NOT RUN (stop rule) |
| P3 | HELD |

Checklist: [x] D0 regression (bitwise solver vs P2-R) [x] D1 folds 20/20 [x] D2 spot audit 30/30
[x] Exp-1 audit PASS [x] final audit PASS [ ] next stage: requires a new human-authorized,
separately frozen pre-registration (none started). Reports `docs/inference_cf/P2_DIR_{EXP1,FINAL}_REPORT.md`.

## P2-SEL — executed (2026-10-06): `P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE`

| Stage | Status |
|---|---|
| P2-SEL implementation + pre-run audit | DONE (`6c7644a`, `c8ec788`, `PASS_TO_P2_SEL_S1`) |
| S1 180-position gate-coupled screen (C0/C1/C2/C3) | DONE, `P2_SEL_AUDIT: PASS (S1)`: D2 benefit retained 78%, margin harm cut 90%, but ZH corruption 5.4% and not non-inferior to BROAD on confusion -> insufficiently selective |
| S2 100-utterance mini free decoding | NOT RUN (stop rule) |
| P3 | HELD |

Next: requires a new human-authorized, separately frozen stage (none started). Report `docs/inference_cf/P2_SEL_REPORT.md`.

## P2-SEL-E — executed (2026-10-06): `P2_SEL_E_DIAGNOSIS_AMBIGUOUS`

| Stage | Status |
|---|---|
| Original freeze / pre-E0 | BLOCKED (`de8d038`, two-token Q premise false); contract repaired at `b693b8d` |
| Implementation + pre-E0 audit r1 | DONE (`0703523`, `7964adb`, `PASS_TO_P2_SEL_E_E0`) |
| E0 diagnosis (no steering) | DONE, `P2_SEL_E_AUDIT: PASS (E0)`: all hypotheses fail -> AMBIGUOUS |
| E1 repair screen / E2 mini decode | NOT RUN (stop rule) |
| P3 | HELD |

## P2-SEL-T — executed (2026-10-06): `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE`

| Stage | Status |
|---|---|
| Implementation + pre-T0 audit | DONE (`d36d7f6`, `50004bd`, `PASS_TO_P2_SEL_T_T0`) |
| T0 token-local diagnosis (no steering) | DONE, `P2_SEL_T_AUDIT: PASS (T0)`: not discriminative |
| T1 causal screen / T2 mini decode | NOT RUN (stop rule) |
| P3 | HELD |

## P2-SEL-XA — pre-outcome design freeze (2026-10-06)

Human-authorized bounded next stage after terminal P2-SEL-E/P2-SEL-T. See
`docs/inference_cf/P2_SEL_XA_SPEC.md`, `P2_SEL_XA_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel_xa.json`.
Exact180 current-query raw-gradient/source compatibility with fixed0.95 scratch check; only supported
audited diagnosis permits clip(S_src,0,1) × E × R_B with unchanged D2/site/alpha/NormPreserve.
Conditional existing100 mini decode only after gate support; no full300 or P3. No scientific run
or runner implementation in design session. Preserve both parents' stops and core-method authority.

## P2-SEL-XA — executed (2026-10-06): `P2_SEL_XA_INVALID`

| Stage | Status |
|---|---|
| Implementation + pre-XA0 audit | DONE (`83845ee`, `e6cf957`, `PASS_TO_P2_SEL_XA_XA0`) |
| XA0 source-compatibility diagnosis | DONE, `P2_SEL_XA_AUDIT: PASS (XA0)`: finite-difference validity failed (4/30 material rows) -> INVALID |
| XA1 gate screen / XA2 mini decode | NOT RUN (stop rule) |
| P3 | HELD |

## P2-SEL-LAC — pre-outcome design freeze (2026-10-06)

Human-authorized bounded next stage after terminal E/T/XA. See `docs/inference_cf/P2_SEL_LAC_SPEC.md`,
`P2_SEL_LAC_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel_lac.json`. Seal unmasked English/Mandarin
next-token candidates; zero exact T-selected waveform W*, re-encode and cold-replay identical prefix.
Only audited support permits E*R_B*max(0,tanh(S_lex/2)) with unchanged D2/site/alpha/NormPreserve;
conditional existing100 mini decode only after gate support. No300/P3. No candidate extraction,
scientific masked outcome or runner implementation in this design session; core authority unchanged.

## P2-SEL-LAC — executed (2026-10-07): `P2_SEL_LAC_NOT_DISCRIMINATIVE`

| Stage | Status |
|---|---|
| Implementation + candidate seal + pre-LAC0 audit | DONE (`4e6062c`, seal, `92b81c6`, `PASS_TO_P2_SEL_LAC_LAC0`) |
| LAC0 forward-only counterfactual | DONE, `P2_SEL_LAC_AUDIT: PASS (LAC0)`: not discriminative |
| LAC1 gate screen / LAC2 mini decode | NOT RUN (stop rule) |
| P3 | HELD |

## P2-SEQ — pre-outcome sequence viability freeze (2026-10-07)

After terminal E/T/XA/LAC, test existing E*R_B*D2 only on the already-frozen100 panel. See
`docs/inference_cf/P2_SEQ_SPEC.md`, `P2_SEQ_CODEX_DESIGN.md`, `configs/inference_cf/p2_seq.json`.
Exactly forced matched no-edit/AUTO/STEER, one allocation max3h. New matched zero-dose required;
AUTO reused after pre-run semantic/hash audit. Development point safety/materiality rules and
independent PASS_TO_P2_SEQ/P2_SEQ_AUDIT frozen. No scientific job or runner in design session.
Stop after100: promising recommends separately frozen confirmation; no gain/damage recommends
evidence-only fresh P2-TTA0 design, never implementation or adaptation here. P3 held; core unchanged.

## P2-SEQ — executed (2026-10-07): `P2_SEQ_SEQUENCE_DAMAGE`

| Stage | Status |
|---|---|
| Implementation + AUTO reuse proof + pre-run audit | DONE (`ed9c355`, `14a2339`, `9cb1d75`, `PASS_TO_P2_SEQ`) |
| Sequence screen (Slurm 57867, 100 utts) | DONE, `P2_SEQ_AUDIT: PASS`: sequence damage; below AUTO |
| P2-TTA0 | evidence-only handoff `docs/inference_cf/P2_SEQ_TTA_HANDOFF.md`; design NOT frozen, nothing run |
| Full 300 / P3 | NOT RUN / HELD |


## P2-TTA0 — pre-outcome episodic objective viability freeze (2026-10-07)

After audited P2-SEQ sequence damage, local steering line stays closed. See P2_TTA0_SPEC/DESIGN,
`configs/inference_cf/p2_tta0.json`, fixed P2_TTA0_PANEL20.json.20 first-in-panel IDs/dialogue,
194 decoder-LN affines only, episodic reset,2 AdamW steps at1e-3, A1 Greedy-EM/A2 AUTO consistency.
Forced-ZH final decoding; no steering, no sweeps. One allocation target<1h/hard2h, independent
PASS_TO_P2_TTA0 and post-audit. Design only, no runner/outcome/job. At most1 viable objective
handoff for separately frozen TTA1; no automatic100/300/P3. Core authority unchanged.

## P2-TTA0 — executed (2026-10-07): `P2_TTA0_INVALID`

| Stage | Status |
|---|---|
| Implementation + pseudo seal + pre-run audit + pushed manifest | DONE (`503c874`, `6d408d2` `PASS_TO_P2_TTA0`, `6a954f5`) |
| TTA0 screen (Slurm 57868, panel20, A1/A2) | RAN; INVALID: frozen live A1 gradient tolerance 1e-3 below bf16 precision (~1e-2); `P2_TTA0_AUDIT: PASS` |
| Evaluation of sealed adapted outputs | NOT RUN (INVALID supersedes; outputs kept sealed) |
| TTA1 / full 100 / 300 / P3 | NOT RUN / HELD; new human decision required |


## P2-TTA-FUNNEL master freeze

P2-TTA-FUNNEL master pre-outcome freeze now defines numerical-only TTA0-R → conditional MAP → selected-objective TTA1. Historical TTA0 INVALID remains. See docs/inference_cf/P2_TTA_FUNNEL_SPEC.md; no funnel outcomes run. Core v6 unchanged.

## P2-TTA-FUNNEL — executed (2026-10-07)

| Stage | Status |
|---|---|
| TTA0-R (repaired audit + sealed evaluation) | DONE: `P2_TTA0_R_OBJECTIVE_SELECTED` (A2), `P2_TTA0_R_AUDIT: PASS` |
| TTA-MAP | NOT RUN (R selected an objective) |
| TTA1 (A2, fixed100, Slurm 57871) | DONE: `P2_TTA1_SUPPORTED` vs forced-ZH; below AUTO on PIER; `P2_TTA1_AUDIT: PASS` |
| Full300 / P3 / transfer / fresh validation | NOT RUN; new human decision required |

## P2-TTA-A3 — executed (2026-10-07): `P2_TTA_A3_TEACHER_UNSAFE`

| Stage | Status |
|---|---|
| Freeze + implementation + `PASS_TO_P2_TTA_A3` | DONE (`bd41bc5`, `eab6c64`, `90dbe96`) |
| A3 24-panel run (Slurm 57876) + seal | DONE (`0d29d35`) |
| Evaluation + `P2_TTA_A3_AUDIT: PASS` | DONE: gap closed, minimal movement, safety failed -> TEACHER_UNSAFE |
| A3-on-100 / tuning / full300 / P3 | NOT RUN; new human decision required |

## P2-TTA-A4 — executed (2026-10-07): `P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED`

| Stage | Status |
|---|---|
| Freeze + implementation + `PASS_TO_P2_TTA_A4` | DONE (`37441eb`, `9488a69`, `d32676b`) |
| A4 24-panel run (Slurm 57937) + seal | DONE (`e506e74`) |
| Evaluation + `P2_TTA_A4_AUDIT: PASS` | DONE: EN transfer + anchor pass; matrix safety not rescued |
| A4-on-100 / tuning / full300 / P3 | NOT RUN; new human decision required |


## P2-PATH0 pre-outcome freeze

P2-PATH0 pre-outcome design frozen: exact12 A3/A4 D rows,5 rescue/7 induction, first-divergence1/3 donor-token clamps under reconstructed unchanged A4. Reference-free released-suffix causal label; secondary ASR after primary seal. One job≤30min later; no run in design session. See docs/inference_cf/P2_PATH0_SPEC.md. A2/A3/A4 remain closed.

## P2-PATH0 — executed (2026-10-07): `P2_PATH0_ASYMMETRIC`

| Stage | Status |
|---|---|
| Implementation + `PASS_TO_P2_PATH0` | DONE (`ce9ff03`) |
| Run (Slurm 58004) + reference-free seal | DONE (`ca4a92e`) |
| Secondary + `P2_PATH0_AUDIT: PASS` | DONE: INDUCE_1/3 pass, RESCUE fail; ASR_HARM_ALIGNMENT INDUCTION_ONLY |
| Branch-aware path control / any follow-on | NOT RUN; new human decision required |


## P2-PATH1 pre-outcome freeze

P2-PATH1 design frozen after PATH0 ASYMMETRIC: U7 state×first-token factorial (six suffix-eligible), exact inherited INDUCE1, fresh same-state caches; fixed H=3 consensus scoring on PATH0 U7/C5. No outcomes in design session. One job≤30min later; see docs/inference_cf/P2_PATH1_SPEC.md. Historical A2/A3/A4/PATH0 unchanged.

## P2-PATH1 — executed (2026-10-07): `P2_PATH1_MIXED`

| Stage | Status |
|---|---|
| Implementation + `PASS_TO_P2_PATH1` | DONE (`cea9410`) |
| Run (Slurm 58011) + reference-free seal | DONE (`2bf92bc`) |
| Secondary + `P2_PATH1_AUDIT: PASS` | DONE: MIXED; CONSENSUS_REJECTS_AUTO PASS; ASR_STATE_ALIGNMENT MIXED |
| Guarded-switch controller / any follow-on | NOT RUN; new human decision required |



## P2-PATH2 pre-outcome freeze

P2-PATH2 design frozen after audited PATH1 MIXED: first online theta0/A4 disagreement, fixed H=3 consensus, G1 one-token branch choice versus G2 short theta0 three-token handoff with clean A4 replay. Exact PATH0 D12; one later job<=30min; no outcome in this design session. See docs/inference_cf/P2_PATH2_SPEC.md. Historical objectives/stages and core v6 unchanged; no automatic follow-on.

## P2-PATH2 — executed (2026-10-07): `P2_PATH2_BRANCH_CONTROL_SUFFICIENT`

| Stage | Status |
|---|---|
| Implementation + `PASS_TO_P2_PATH2` | DONE (`155aa40`) |
| Run (Slurm 58013) + reference-free seal | DONE (`e1198aa`) |
| Secondary + `P2_PATH2_AUDIT: PASS` | DONE: G1 = G2 pass (ZH 21 / POI 101 / mixed 123); no G2 advantage |
| Broader G1 development test | NOT RUN; separately frozen contract and human decision required |



## P2-PATH3 pre-outcome freeze

P2-PATH3 pre-outcome design frozen after audited PATH2 BRANCH_CONTROL_SUFFICIENT: exact unchanged A4+G1 on historical fixed100, replay PATH2_12 first before remaining88; NOVEL76 rescue breadth/concentration required. NOVEL76 is already TTA1-exposed D-dev-select, NOT fresh validation. One later job<=30min; no scientific outcomes in design session. See docs/inference_cf/P2_PATH3_SPEC.md. No G2/controller tuning/full300/P3; core v6 unchanged.

## P2-PATH3 — executed (2026-10-07): `P2_PATH3_SIGNAL_CONCENTRATED`

| Stage | Status |
|---|---|
| Implementation + `PASS_TO_P2_PATH3` | DONE (`ddfa787`, `71a51a1`) |
| Manifest + run (Slurm 58015) | DONE (`cd93f3d`; replay12 exact barrier, then 88) |
| Reference-free seal + primary audit | DONE (`bceb341`, `77f4ee9`) |
| Secondary + `P2_PATH3_AUDIT: PASS` | DONE: safety/rescue/gain/retention pass; NOVEL76 breadth fails (A4 inert there: native LID = zh no-op) |
| Further G1 breadth test | NOT RUN; needs a population where A4 activates, a separately frozen contract and a human data-role decision |

## P2-PATH4 pre-run design — BLOCKED

Frozen intended unchanged G1 transfer to exact historical A2 on fixed100; no runner/scientific job. Reference-free token inspection reproduces A2_DELTA14 (7 outside DEV24), but four theta0-EOS/A2-content first disagreements violate the inherited EOS-first INVALID rule. No PASS_TO_P2_PATH4 or execution authorization. See docs/inference_cf/P2_PATH4_SPEC.md. A separate authorized boundary revision is required; no row exclusion/fallback/controller tuning. Core v6/history unchanged. P3 HELD.

## P2-PATH4-R1 — authorized EOS-boundary extension

Original P2_PATH4_BLOCKED_EOS_FIRST_BRANCH (29661e9) preserved. Human authorized PATH4_R1_EOS_BOUNDARY_V1 before any scientific run/PATH4 reference outcome. Content/content G1 unchanged; first EOS/content disagreement uses equal-horizon1 next-action .5/.5 consensus, inherited tie and one-action A2 execution. Fixed100/A2/thresholds/firewall unchanged. Additive eos_boundary helper, CPU tests and independent freeze auditor; no runner/scientific outcomes/GPU. See docs/inference_cf/P2_PATH4_EOS_AMENDMENT.md. PASS_TO_P2_PATH4_R1 required before later execution; runnable manifest/code audit remains mandatory. P3 HELD.

## P2-PATH4-R1 — executed (2026-10-07): `P2_PATH4_OVERCONSERVATIVE`

| Stage | Status |
|---|---|
| Revision-0 freeze | BLOCKED_PRE_RUN_EOS_FIRST (`29661e9`, preserved) |
| R1 amendment + freeze audit | DONE (`490ec20`, `PASS_TO_P2_PATH4_R1` freeze scope) |
| Implementation + runnable `PASS_TO_P2_PATH4_R1` | DONE (`bd5db02`, `316b727`) |
| Manifest + run (Slurm 58035) | DONE (`c57936a`; all-100 A2 barrier exact, then R1 G1) |
| Reference-free seal + primary audit | DONE (`4781605`, `fcf000f`) |
| Secondary + `P2_PATH4_AUDIT: PASS` | DONE: safety/rescue pass; benefit retention fails (0.724, PIER +0.0115); breadth fails |
| Any EOS-boundary redesign | NOT RUN; needs a separately frozen contract and human decision |

## P2-OPP0-ABSTAIN + P2-PATH5 freeze (2026-10-08)

OPP0 (no GPU): derived A2+G1A on fixed100 from sealed PATH4 rows: ZH 1093 / POI 314 / mixed 1439 (A2 1116/316/1464); EOS harm removed,
content G1 preserved, rescue still only PATH2_12 (`docs/inference_cf/P2_OPP0_ABSTAIN_REPORT.md`). PATH5 frozen pre-outcome: A2+G1A
(content/content original G1; EOS boundary -> abstain to A2) on NEW200 = FULL300 - fixed100 (already-exposed D-dev-select, not fresh),
fixed100 live barrier first, reference-free opportunity gate (>=3 changed rows and dialogues), 0.90 strong retention, breadth; hard stop:
anything but G1A_SUPPORTED closes the controller line and returns to A2. See `docs/inference_cf/P2_PATH5_SPEC.md`.

## P2-PATH5 — executed (2026-10-08): `P2_PATH5_SAFE_NO_ADDED_VALUE` (G1 controller line closed)

| Stage | Status |
|---|---|
| OPP0-ABSTAIN derived diagnostic | DONE (`7745e5b`, `d1d057d`) |
| PATH5 freeze | DONE (`d1d057d`) |
| Implementation + `PASS_TO_P2_PATH5` | DONE (`14bd563`, `01d57b8`; regression 91/91) |
| Manifest + run (Slurm 58074) | DONE (`48f0ace`; fixed100 barrier exact, NEW200 executed) |
| Seal + primary audit | DONE (`101ca00`, `e1086f2`; opportunity gate open 7/6) |
| Secondary + `P2_PATH5_AUDIT: PASS` | DONE: safety + retention pass, R_net -2 -> SAFE_NO_ADDED_VALUE |
| Decision | G1/controller line CLOSED; main adaptation candidate = A2; next = upstream adaptation/objective (separately frozen) |
