# P2-TTA-FUNNEL — pre-outcome master contract

This is a separate, already-exposed D-dev-select development program, not a revision of the core v6 method. Starting local and remote HEAD: `e397d38c08bc8b0670a27fb6078ae5364878897b`; clean checkout, no pending/running user jobs. The historical P2-TTA0 result remains **P2_TTA0_INVALID**. No adapted reference outcomes have been inspected for this freeze. The only new inspection was theta0 baseline transcript equality to freeze MAP selection. Historical steering and selectivity stages remain closed.

Authority: this spec and `configs/inference_cf/p2_tta_funnel.json`; panel-selection and inheritance contracts linked below are normative. Original TTA0 config is byte-hash anchored. Conflicting resolved settings fail pre-audit; do not infer alternative scientific choices.

## Branches and stop conditions

1. TTA0-R audits sealed run1 with the numerical correction below. Failed audit → **P2_TTA0_R_STILL_INVALID**, STOP entire funnel.
2. Only after **PASS_TO_P2_TTA0_R_EVALUATION**, evaluate run1 using the original TTA0 scientific rules. Independent post-audit required. Exactly one selected objective → immutable selection seal, then TTA1. Neither viable → **P2_TTA0_R_NO_VIABLE_OBJECTIVE**, then MAP.
3. MAP runs only after that audited no-viable label. Invalid or no viable objective stops the entire funnel. Exactly one selected objective → immutable selection seal, then TTA1.
4. TTA1 runs once for that selected formulation. Stop after its report regardless of result. No full300/P3 or further LR/steps/objective/subset exploration.

No later stage starts until its predecessor report, independent PASS and immutable selection artifact (where needed) are committed and pushed. No stages run in parallel. Outcome reports are not created in this design session.

## A: TTA0-R — numerical repair and sealed-output evaluation

Only override the independent complete-master-gradient criterion:

`||g_primary-g_auditor|| <= max(1e-8, 0.02 * ||g_auditor||)`.

The old relative tolerance was 0.001. Keep absolute loss agreement **1e-5**, all finite-gradient checks, float64 equivalence, deterministic repeated fp32 equivalence, exact trainable identity, reset, output and manifest checks unchanged. Justification is exclusively the existing reference-free `run1_live_audit_diagnostic.json`: recorded CPU mixed-precision relative discrepancies 0.749–1.044%, live A1 discrepancy 1.025%, exact float64 primary/auditor gradients and repeated fp32 gradients. The loss agreement check applies to the sealed live run values, not a newly substituted CPU implementation. This bounded numerical allowance does not excuse a formula/mask/trainable mismatch.

The config freezes every run1 row, both objective checkpoints, theta0 LN snapshot, manifest/runtime, pseudo seal, original audit, invalid record and numerical diagnostic by SHA256. Both A1 and A2 outputs are contained in each sealed row. Baseline20 paths/hashes and panel20 hash are frozen too. Verify all bytes before opening reference data. Any mismatch → STILL_INVALID. Preserve original invalid audit/report/config. Write a new R audit/report; do not overwrite them.

**No new optimizer step, adaptation backward, pseudo teacher, generated transcript, or parameter update.** CPU inspection of recorded independent-gradient evidence and audit recomputation of saved vectors/scalars are allowed. Audit revalidation uses existing recorded live objective checks and numeric diagnostic, not a new model adaptation run. Recompute saved update/reset/count mechanics where possible. If an original required check cannot be established from sealed evidence, STOP; do not rerun adaptation to manufacture evidence.

After repaired audit PASS, evaluate B0-FORCED, B0-AUTO, A1 and A2 with the exact original TTA0 canonical metrics, viability, confirmation-bias, bootstrap and objective selection. `original_tta0_decision` is an exact copy of the original config's decision section:

- Safety: MER increase ≤0.01; ZH-CER increase ≤0.015; matrix-ZH retention ≥0.98; embedded-EN retention ≥0.95; outside-POI lexical harm ≤0.03; POI corruption ≤0.05; additional caps ≤1; new severe truncations =0.
- Benefit: net POI error reduction ≥2 **OR** at least3 utterances have fewer mixed errors and improved count strictly exceeds degraded count.
- Severe truncation: B0 content length≥10 and adapted EOS content length≤floor(0.5×B0 length). Empty baseline-correct rate denominators are None/not assessable with vacuous bound; primary reference language populations must be nonempty. Float guard1e-12.
- A1 confirmation bias: fixed-y_B token-weighted entropy decrease≥1% AND any original safety bound fails. No safety rule is relaxed by entropy decrease.
- Select only viable objectives; if both, lower MER, PIER, ZH-CER, mean per-utterance fp32-master update L2; ties within1e-12 proceed to next component, final exact tie A2.

Stage labels: STILL_INVALID first, then **P2_TTA0_R_OBJECTIVE_SELECTED**, otherwise **P2_TTA0_R_NO_VIABLE_OBJECTIVE**. Per-objective original TTA0 labels are retained. Required post-audit: **P2_TTA0_R_AUDIT: PASS**.

## Common adaptation mechanics for MAP and TTA1

Inherit the hash-locked TTA0 trainable list: decoder LayerNorm affine parameters only,194 tensors/248320 scalars. All96 decoder sublayer LNs plus final decoder LN; encoder and all non-LN decoder parameters stay frozen. Fresh fp32 master parameters and fresh AdamW each utterance/objective: lr1e-3, weight_decay0, betas(0.9,0.999), eps1e-8, exactly2 steps, no clipping/scheduler/accumulation/scaler/autocast; eval mode. Resident forward bf16/eager through differentiable cast/functional_call; master updates fp32. Exact theta0 reset and no caches/optimizer/grads transferred between episodes. Detached theta0 encoder once per utterance; fresh decoder KV caches per final decode. Parameter list, masks and generation settings must match original config.

Greedy, batch1, max_new_tokens200, no sampling/beam. Final decoding always forced-ZH through the original unsteered cached path. No steering/D2/native-LID/router calls. A passive capture module in the historical decode path must not edit states. Teacher forcing uses no KV cache, fixed content tokens, excludes prefix and EOS as targets; EOS remains a predictive class. Canonical suppressed vocabulary and begin-suppression at first content query are identical for teacher/student; valid mask is fixed before updates. An empty valid teacher follows original TTA0: exactly2 zero-gradient/no-op AdamW steps with a scalar zero connected to masters, baseline-identical final decode, no drop/refill. It is not informative for optimization. Nonfinite loss/gradients, changed parameter identity or reset failure is INVALID.

## B: TTA-MAP

Run only if R selects nothing. Panel: [selection contract](P2_TTA_MAP_PANEL_CONTRACT.md), [frozen24](P2_TTA_MAP_PANEL24.json). The stored theta0 100-row baselines contain12 disagreements across9 dialogues and88 agreements. Frozen MAP has all12 disagreements plus12 agreements,24 utterances/20 dialogues. Selection depends solely on exact stored text equality and parent order, never references or adapted outputs.

Exactly two objectives:

- **A1 GREEDY-EM**: identical TTA0 entropy minimization on frozen theta0 forced greedy y_B.
- **A3 SOFT-AUTO-KL**: frozen theta0 AUTO greedy y_A, identical y_A content prefix path for teacher/student. At temperature1, full allowed vocabulary: `q_t=softmax(z_theta0,AUTO,t)`; `p_t=softmax(z_theta,forcedZH,t)`; `L=mean_valid_t sum_v q_t(v)*(log q_t(v)-log p_t(v))`. Teacher q and logq are float32, detached, computed once before updates. No top-k, hard NLL, entropy mix, regularizer or lambda.

Recover actual theta0 AUTO prompt with installed Whisper4.57.6 `detect_language` on detached encoder outputs and the historical generation config: SOT/native selected language/transcribe/no-timestamps. Native language may be ZH: never substitute EN or another condition. Check teacher prompt/provider/model/config hashes and theta0 AUTO output provenance. If AUTO and forced conditions coincide, initial D_cond can be zero; report this as absence of conditioning gap, not INVALID. No new teacher transcript search.

Both objectives log L0/L1/L2, global pre-step gradient norms, fp32/effective bf16 update norms, common fixed-y_B entropy/script mass/EOS signals, final forced decode, lengths/caps/truncation and token edit distances to both baselines. At initial/step1/step2 compute `D_cond=mean_valid KL(q_AUTO||p_FORCED)` on fixed y_A for both objectives; for A3 it is the objective, for A1 descriptive. Log runtime, allocated/reserved VRAM, forwards/backwards. Seal complete outputs/checkpoints before opening references.

### Exact MAP diagnosis and advancement

All aggregate losses/gaps are token-count-weighted sums over fixed valid positions. A row is objective-informative if L0≥1e-4. `optimized` requires ≥6 informative rows, aggregate (L0−L2)/L0≥0.10, and ≥ceil(0.60×informative rows) individually decrease≥10%. Apply relative thresholds only to informative rows; empty/zero diagnostic populations report not assessable. A1's intended diagnostic is its entropy decrease; A3's is D_cond decrease.

`gap_closed`: D-group token-weighted D_cond0≥0.01 nat and relative gap reduction≥10%. `sequence_movement`: at leastceil(0.25×n_D) D utterances change content token sequence versus FORCED. `toward_AUTO`: at leastceil(0.25×n_D) D utterances strictly reduce token Levenshtein distance to AUTO AND summed D distance to AUTO decreases≥10%. No normalized/text quality score chooses rows.

The10% response and25% transcript-movement criteria are fixed materiality screens, well above the recorded roughly1% numerical discrepancy; they are not fitted to adapted outcomes. Six informative rows prevents a few nonzero-gap cases from dominating.

`safe` and `useful` use the original TTA0 point safety and benefit rules above, applied to all24; no significance gate. A1 `promising` = optimized AND safe AND useful. A3 `promising` additionally requires gap_closed, sequence_movement and toward_AUTO, and n_D≥6.

Compute diagnostic flags independently, then choose primary label in this order:

1. Engineering failure → **P2_TTA_MAP_INVALID**, stop.
2. Not optimized → **OBJECTIVE_NOT_OPTIMIZED** (record any damage as secondary).
3. Unsafe candidate with any new severe truncation, >1 extra cap, or increased deletion errors/reference mixed-unit count >0.01 accompanied by total generated content-length ratio<0.90 → **SEQUENCE_INSTABILITY**. Also use this label for other unsafe candidates not meeting the teacher-attribution condition below, identifying the violated bound explicitly; it means observed sequence safety failure, not proven causal mechanism.
4. Promising → **OBJECTIVE_PROMISING**. A safe useful student can advance even when AUTO itself is unsafe.
5. Unsafe candidate, toward_AUTO true, and AUTO violates at least one of the same outcome safety bounds versus FORCED → **TEACHER_NOT_SAFE**. Report association, not causal proof. Strong truncation/deletion instability in3 takes precedence.
6. Not promising and gap_closed false → **CONDITION_GAP_NOT_CLOSED**.
7. Remaining non-promising cases → **SEQUENCE_LEVERAGE_LIMIT**: optimization/gap response did not produce enough *useful safe* output movement. Report whether movement was absent, away from AUTO, or present without canonical benefit.

If total parent D<6, mark LOW_DISAGREEMENT: no strong conditioning-gap/teacher/leverage attribution; A3 cannot advance. For non-promising rows retain labels with LOW_DISAGREEMENT qualifier and descriptive interpretation only. A1 can advance via its ordinary optimized/safe/useful rule. This contingency is frozen but false for this panel.

Only OBJECTIVE_PROMISING may advance. If both, lower corpus MER, PIER, ZH-CER, outside harm, mean fp32-master update L2; component ties1e-12, final tie A3. At most one. Labels **P2_TTA_MAP_OBJECTIVE_SELECTED / P2_TTA_MAP_NO_VIABLE_OBJECTIVE / P2_TTA_MAP_INVALID**. Pre-gate **PASS_TO_P2_TTA_MAP** and post **P2_TTA_MAP_AUDIT: PASS**.

## C: TTA1

Run only with one independently audited selection seal from R or MAP. See [inheritance contract](P2_TTA1_INHERITANCE_CONTRACT.md). Exact parent100 hash `266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`; no reselection/refill/drop. Systems only B0-FORCED, B0-AUTO, SELECTED-TTA. Same exact objective/teacher/capacity/masks/precision/reset/settings as selected; A2 is allowed only through R, A3 only through MAP. No second objective.

Reuse hash/semantic-audited theta0 baselines. Reuse all compatible independently audited selected-objective ancestor episodes; choose MAP before R if both exist, never choose by outcome. If not compatible, new episode in the one TTA1 allocation. Every parent ID must have exactly one resolved output with provenance. No hybrid settings across rows.

Canonical PIER/MER/EN-WER/ZH-CER; matrix-ZH/embedded-EN retention; POI corrections/corruptions; outside lexical harm; substitutions/deletions/insertions; caps/severe truncations. Loss/update response, output-change fraction, runtime, VRAM and backwards; A3 D_cond, A1 entropy/confirmation-bias, A2 NLL. Compare SELECTED-TTA to FORCED causally and separately AUTO practically. Never redefine the causal baseline using AUTO.

Decision precedence:

1. Missing/wrong panel/output/provenance/settings/evaluator/nonfinite/reset/audit → **P2_TTA1_INVALID**.
2. Any original TTA0 safety point bound fails → **P2_TTA1_SEQUENCE_DAMAGE** (same caps≤1 and new severe truncations0 on100).
3. Safe AND net POI errors reduced≥5 AND absolute PIER decrease≥0.005 → **P2_TTA1_SUPPORTED**.
4. Otherwise → **P2_TTA1_NO_USEFUL_GAIN**.

This is exposed development replication, not fresh-validation confirmation. Confidence intervals are descriptive, not a significance gate. No utterance-count benefit alternative at100. Report AUTO deltas even if supported against FORCED. Stop and handoff; no full300/P3 or automatic subsequent experiment.

## Uncertainty, integrity, compute and firewall

All stages reuse deterministic dialogue-block bootstrap:2000 paired draws, seed240924, sorted dialogues, corpus count-sum ratios,95% pointwise percentile; omit/disclose zero-denominator draws. Independent auditor recomputes metrics, masks, selections and bootstrap; must not import primary decision code. Canonical metric primitives may be shared. MAP/TTA1 independent first-row loss/complete fp32-gradient checks use abs loss1e-5 and relative gradient2%, floor1e-8; test A3 independent float64 identity and fp32 repeat before run. Numerical failures stop, no tolerance retuning.

TTA0-R:0 new scientific GPU jobs; MAP≤1 job24×2×2=96 updates, target<15min, hard60min; TTA1≤1 job≤200 updates, target<1h, hard3h. Sequential sbatch, one H100 MIG3g.40gb allocation per stage, at most one pending/running job. Wall-limit exhaustion seals partial INVALID and stops; no extra jobs. No high-dimensional teacher/activations committed; transient full vocabulary teacher arrays are allowed, tiny LN checkpoints saved for independent audit.

Allowed only exposed D-dev-select fixed100, sealed run1 and inference_cf artifacts. Forbidden router-calib new role, D-dev-confirm, D-test, full300, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND, transfer/fresh validation; no steering/continual state. Reference/evaluator access only after stage construction and output seals pass. MAP transcript equality is theta0-only, not outcome quality.

Before each new outcome: tests, independent pre-audit, resolved manifest through existing provenance utilities, teacher/selection seals, code/config/git/model/environment/input hashes; commit/push. After outcome: independent audit PASS before any scientific conclusion/branch; commit/push reviewed artifacts. No force push, main merge or PR. This design session launches no scientific jobs and evaluates no adapted references.
