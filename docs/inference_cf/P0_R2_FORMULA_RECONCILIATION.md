# P0-R2 formula reconciliation — final design contract

**Status: `FORMULA_RECONCILED; R2_FEASIBILITY_SPEC_PENDING`.** Scientific design only, 2026-09-24. This is a separately versioned inference-time proposal. It does not revise the repository's DG-03R core method, the P0/P0-R1 frozen evidence, or the independent `P0_R2_DESIGN_AUDIT.md`. No R2 GPU result, intervention, or P1 claim exists. The next spec must freeze the implementation and decision criteria below before execution.

## Evidence and baseline decision

Read with `INFERENCE_STEERING_IMPLEMENTATION_PLAN.md`, `FEASIBILITY.md`, `P0_INDEPENDENT_AUDIT.md`, `P0_FEASIBILITY_SPEC.md`, `P0_R1_EVIDENCE_SPEC.md`, `P0_R1_REPORT.md`, the independent R2 audit, and current `METHOD_CONTRACT.md`, `STATUS.md`, `CODE_MAP.md`, and `DATA_EXPOSURE.md`. P0/R1 established `cB=cM` and bit-identical `hB=hM` for forced Chinese, valid same-prefix replay, and failed K=1/K=3 candidate-continuation support. They did **not** validate a steering direction or gate. The R2 audit additionally found a strong English preference on a zero waveform (raw pairwise E ≈ 0.94), low-classified-mass degeneracy in conditional conflict, and a label provenance mismatch between auto-LID POIs and forced-ZH decodes.

**Frozen deployment assumption:** M and E are supplied task metadata. For the primary Mandarin–English Whisper experiment, `cB=cM=forced zh` (`[50258,50260,50360,50364]`), so `hB=hM` is an expected computational identity. `B0` is ordinary forced-ZH decoding. `B0_AUTO` (ordinary unsteered auto-LID decoding) is a **mandatory comparator**, including its PIER, MER, English and Mandarin errors, and relevant confusion categories. Its stronger development PIER in the audit (0.390 vs B0's 0.470) must remain visible. The paper must answer both whether the method improves B0 in the known-M setting and how it compares with simply enabling Whisper auto-LID. No result on B0 alone establishes superiority to B0_AUTO. This choice is closed for R2.

## 1. LocalSupport: `LOCALSUPPORT_FREEZE = LS-B`

Let `pi_t(L)` be Whisper's native language-token softmax on a zero-padded 30 s input containing the causal local audio window. Let `pi_null(L)` be the same provider, model, precision, and preprocessing on an all-zero 30 s waveform. Fix `eps=1e-12` and compute

```text
l_t    = log((pi_t(E) + eps) / (pi_t(M) + eps))
l_null = log((pi_null(E) + eps) / (pi_null(M) + eps))
A_t    = l_t - l_null
E_t    = max(0, tanh(A_t / 2))
       = 2 * max(0, sigmoid(A_t) - 0.5)
```

`LocalSupport` means **positive excess English evidence relative to this model's null/template odds**, scaled to `[0,1)`. It is neither a calibrated probability that the segment is English nor a physical acoustic likelihood ratio. If `Lambda=exp(A_t)` is the *ratio of model odds* to null odds, then `E_t=[(Lambda-1)/(Lambda+1)]_+`. Thus `Lambda<=1` gives exactly zero; increasing positive evidence gives a bounded monotone strength. This is a defensible positive odds-ratio margin without claiming generative likelihood semantics.

LS-A (`sigmoid(A_t)`) assigns `E=0.5` when the window supplies **no more English evidence than the null template**. Because Mandarin positions often have high conflict, the product would then be `g≈0.5R`, a substantial activation floor on precisely the positions needing protection. LS-A's neutral value is coherent for a *pairwise probability-like* score, but such a probability is neither calibrated nor the intended gate semantics. LS-B deliberately discards negative values **from the action strength**; retain signed `A_t`, both posteriors, pair mass, and the unclipped sigmoid in diagnostic rows. Negative evidence can rank Mandarin windows for analysis, but it must not produce negative steering or residual positive action. No third formula is needed.

For `A>0`, LS-B is a strictly increasing transform of LS-A, so positive-region ordering is the same. LS-A ranks negative windows whereas LS-B ties them at zero; AUROC can therefore change, especially with many negatives. The product ranking can change more because LS-A's 0.5 offset interacts with conflict. The absolute gate dose also changes and **cannot be reused** from LS-A or tuned from R2 outcomes. Later dose selection, if R2 succeeds, belongs to the predeclared development protocol.

## 2. BaselineConflict: `BASELINECONFLICT_FREEZE = BC-B`

From unsteered forced-ZH next-token logits, upcast to float32 before `log_softmax`, and sum disjoint, frozen tokenizer classes:

```text
P_M(t) = sum_{v in V_M} pB_t(v)
P_E(t) = sum_{v in V_E} pB_t(v)
Q_t    = P_M(t) + P_E(t)
R_t    = max(0, P_M(t) - P_E(t))
```

`V_M` includes pure-Han and Han-lead UTF-8 (`0xE4–0xE9`) fragments; `V_E` includes ASCII-Latin word tokens. All other tokens, including continuation-only bytes, mixed script, punctuation, digits, EOS and controls, remain ambiguous and are logged by class. Freeze the exact tokenizer revision and partition hash. These classes indicate **script affinity**, not lexical Chinese/English; the provider does not transfer to same-script pairs without a new contract.

`BaselineConflict` represents **excess matrix-script probability mass in the full next-token distribution**: the baseline is allocating probability to an M-like continuation over an E-like one. For `Q>0`, ignoring epsilon, `R_mass=Q*R_cond`; also `0<=R_mass<=Q`. Therefore as `Q→0`, conflict necessarily vanishes continuously. BC-A's conditional margin can approach one while both masses approach zero, and its `Q>=0.5` rule is a discontinuity at an unvalidated development threshold. A majority of full-vocabulary mass is not a mathematical requirement for a meaningful repair conflict.

BC-B does downweight a strong *conditional* M preference when punctuation or neutral tokens take much mass. That is appropriate here: the actual decoder has less M-like probability available to redirect at that step. This can miss a deletion before punctuation or EOS; R2 must expose that limitation rather than lower a threshold after seeing it. It is not a proof that a token would be repaired. `Q`, `P_M`, `P_E`, ambiguous mass, and the conditional margin remain diagnostics. There is **no hard low-Q abstention**. `Q` near zero causes `R` and `g` near zero by construction.

On Mandarin and Han-rendered English confusion, `R` is typically high only when M mass is substantial. Correct English usually has `R=0` or small. Same-language English substitutions usually have low `R` and are outside this mechanism. Mixed-script and punctuation-adjacent positions may have intermediate/small `R`; EOS often has small `Q`. A deletion has no emitted word: evaluate its next-token **gap slot**, including an EOS slot when present, and report the resulting `Q/R/g` separately. Do not hard-veto EOS merely because it is a special token: that would silently rule out end deletions. The reference-free gate itself never sees the deletion label.

## 3. Product behavior and claim boundary

```text
g_t = E_t * R_t if structurally eligible; otherwise 0.
```

| Case | Expected factors and action |
|---|---|
| Correct Mandarin | `E≈0`, `R` often high ⇒ `g≈0`; localization bleed or acoustic/LID failure remains an empirical false-positive risk. |
| Correct English | `E>0`, `R≈0` ⇒ `g≈0`. |
| English→Han or phonetic transliteration/script conversion | `E>0`, `R>0` ⇒ high relative `g` when the causal window covers the English audio. |
| English→different English word | `R≈0` ⇒ low `g`; out of mechanism scope. |
| English deletion | No matching emitted token; use an explicit next-token/EOS gap slot. It is repairable by this gate only if the slot's causal window still covers English and `R` has meaningful M excess. |
| Padding/no speech | `A≈0`, hence `E≈0`, `g≈0` for the identical null template. This does not establish that every silence or noise window behaves identically. |
| Low classified mass | `R<=Q`, hence `g<=Q`; no cliff or artificial near-maximal conflict. |

`g` is a **repair-need ranking/strength score, not a calibrated error probability**. A high gate is not proof an edit helps. The primary mechanistic claim is **inference-time repair of embedded-language errors associated with matrix-language confusion**. For this pair, primary targets are `wrong_language_substitution` and `phonetic_transliteration_or_script` (`EN-confusion`). `EN-deletion-slot` is secondary and conditional. `EN-same-language-sub` is out of mechanism scope but remains in overall PIER/WER and is never called a successful correction. Preserve `EN-correct`, `ZH-correct`, and explicit `unalignable:*` strata. Overall recognition and damage metrics count all errors.

## 4. Causal localizer and structural eligibility

At the query predicting `y_t`, use only `(x,cB,y_<t)`. Read the shipped `alignment_heads` attention probabilities for that query, average heads, zero frames outside the heard audio, renormalize over `F=min(1500,ceil(min(duration,30)/0.02))` valid encoder frames, and take the earliest `argmax` frame `c_t`. The primary 1.0 s waveform window is centered at `0.02*c_t` seconds, shifted (without shortening) inside `[0,min(duration,30)]`; if seen audio is <1.0 s, use all seen audio. Crop the actual waveform at 16 kHz sample boundaries, then use the frozen Whisper preprocessing and pad to 30 s for native LID. The R2 spec must fix frame/sample rounding and save exact indices. A full causal teacher-forced replay of the **fixed baseline output** is equivalent for each query; it must not use future tokens in the score calculation. Never use post-hoc Whisper DTW or reference/CTC timings in inference.

**One primary window rule only.** The audit's possible 2.0 s fallback triggered by pair mass below the null-template value is rejected: that value is not a technical invalidity boundary, and changing window size would change the provider. A failed/undefined 1.0 s window abstains with a logged reason. No window search from gate labels or AUROC.

Structural fallback, with deterministic reason precedence fixed in the R2 spec:

```text
prefix ends in incomplete UTF-8 content bytes -> g=0, mid_character
no valid current-query/head attention or audio frame -> g=0, localizer_fail
nonfinite logits, posterior, null odds, mass, score -> g=0, nonfinite_signal
```

The prefix test uses raw tokenizer bytes, not replacement-character decoded text. Normalization failure or missing local audio similarly records `localizer_fail`; provider failure records a specific reason. No low-Q structural abstention exists. Log every fallback and keep those positions in the dense panel and end-to-end denominators. A zero/tiny direction norm is a later P1 no-edit fallback, not an R2 gate-label exclusion. The R2 spec must define any further purely technical invalidity before data execution.

## 5. Evaluator-only alignment and panel

Recompute categories from the **exact forced-ZH R2 baseline** and canonical PIER reference↔hypothesis alignment; never reuse P0's auto-LID-derived labels or coarse character-fraction positions. Preserve generated token IDs and map normalized hypothesis units to the query predicting their first token through token/character offsets and a checked normalization offset map. Substitutions/correct units map to that token query; deletions map to the next aligned hypothesis unit's query or EOS gap slot. Consecutive deletions sharing a slot are not independent position observations. Mark offset collisions, ambiguous maps, boundary errors, targets beyond heard 30 s, and truncation explicitly `unalignable:*`; never force a target onto a convenient token. A position with conflicting stratum labels is excluded from primary AUROC with a reason, while its underlying reference units stay in overall metrics.

Build a **versioned dense R2 panel** over the existing 300-utterance D-dev-select population (20 dialogues), with all generated content positions and explicit gap slots, regardless of gate values. Mapped reference units supply diagnostic strata. Evaluator-only MMS-FA/CTC timings may validate the causal localizer; they never provide windows or inference inputs. The predeclared audio control replaces only the local waveform by the deterministic cyclic-permutation donor's audio at the same window seconds (with a predeclared donor-boundary rule); keep position, prefix, logits, and `R` fixed. `B0_AUTO` is a separate unsteered comparator on the same utterance IDs. No D-dev-confirm/D-test result enters selection; no P1 intervention is run.

## 6. Recommended pre-run feasibility criteria

The next R2 spec must freeze the exact panel, serialization, numerical implementation, and these rules **before any R2 GPU outcome**. A threshold here is an operational feasibility bar, not a biological or probabilistic constant. Report distributions, failure reasons, denominators, and dialogue-cluster uncertainty even when a criterion fails. The provider-coverage denominator is every baseline step with a complete UTF-8 content prefix and valid heard audio **before** attempting localization/LID; provider failures count against it. For **gate** AUROC, use every uniquely mapped decode position, including structural fallbacks at `g=0`; never improve the metric by dropping abstentions. Avoid duplicate reference units at one slot. Require at least 30 mapped positions **and 10 dialogues in each compared stratum**. If absent, the relevant contrast is *not estimable*, never a pass. Report 95% dialogue-cluster bootstrap percentile intervals using 2,000 dialogue resamples and seed 240924.

| Check | Frozen recommended pass rule | Rationale/status |
|---|---|
| `ALIGN` | ≥80% of non-boundary English POIs map to a unique token position or gap slot; all others counted by explicit reason. | Engineering coverage floor; inherited from audit and sufficient to detect unusable mapping, not a proof of no alignment bias. |
| `LOCALIZER` | ≥60% of mapped units with valid evaluator-only CTC midpoint have a valid center and `|0.02c_t-midpoint|≤0.5 s`; localizer fallbacks count as misses. Report median absolute error and actual-window midpoint coverage by stratum. At least 30 timed units and 10 dialogues in English and Mandarin strata; otherwise not estimable. Fail blocks LS, with no window search. | `0.5 s` corresponds to the nominal half-window; `60%` is an engineering sanity floor, **not** scientifically derived or a claim of high-quality localization. |
| `R2-LS` | Finite `E` on ≥90% of the pre-provider coverage denominator; on finite, uniquely mapped LS positions, `AUROC(E; EN-correct ∪ EN-confusion versus ZH-correct)≥0.70`, with cluster-bootstrap lower 95% bound >0.5. Paired real/mismatched-audio coverage must be ≥90% of those finite mapped LS positions; on that paired subset, the lower 95% dialogue-cluster bound for `AUROC_real−AUROC_mismatch` must be >0. Report exclusions by stratum. Verify identical null input gives `A=0,E=0` to numerical tolerance (`1e-6`). | 90% coverage and 0.70 are predeclared engineering/effect-size bars. The null identity and positive real-audio advantage test the claimed acoustic contribution without an arbitrary raw-score-change cutoff. Save IQR and pair mass descriptively. |
| `R2-BC` | Every step in the pre-provider coverage denominator has finite `P_M,P_E,Q,R` within probability bounds, disjoint frozen classes, `R=[P_M-P_E]_+`, `0≤R≤Q`, and `R=0` when `P_M≤P_E`; zero numerical invariant violations beyond `1e-6`. Report sign concordance and `Q/R` by every stratum, including EOS/gaps. | Mathematical/engineering validity. No arbitrary `R≥0.5` 90% rule; `0.5` is not a semantic boundary for mass difference. Discrimination is assessed by the combined gate. |
| `R2-GATE` | `AUROC(g; EN-confusion versus ZH-correct)≥0.70` **and** `AUROC(g; EN-confusion versus EN-correct)≥0.70`; each lower 95% dialogue-cluster bound >0.5. Report `EN-deletion-slot` secondarily, Mandarin near-English false activation, gate quantiles, and unalignable/fallback counts. | These are predeclared practical effect-size bars, not a calibrated probability threshold. The first tests the substantive confusion-vs-Mandarin question; the second checks protection of correct English. |

The audit's `IQR(E)≥0.10`, mean `|ΔE|≥0.05` audio-control pass rule, `R≥0.5`/`R≤0.5` on 90% of English strata, and `g≥0.5` false-activation cutoff are **rejected as pass rules**: they impose arbitrary scale cutoffs, can reward irrelevant audio sensitivity, or become invalid under the one-sided/mass formulas. The audio control instead requires a positive *paired AUROC advantage* with a chance-excluding cluster interval; raw paired score changes remain descriptive. Record IQR, conditional margin, `Q`, and score quantiles. The audit's AUROC 0.70 is retained as a predeclared operational effect size, strengthened with a chance-excluding cluster interval. Thirty positions alone is too weak for dialogue-level inference, so add the 10-dialogue minimum and uncertainty condition. Neither threshold nor provider may be retuned from R2 outcomes.

**Decision:** `R2_FEASIBLE` requires ALIGN, LOCALIZER, R2-LS, R2-BC, and both R2-GATE contrasts to pass with no inference leakage or provenance failure. Otherwise `R2_BLOCKED` with the failed component. Do not start P1 on a partial pass.

## Compact implementation contract

```text
LOCALIZER:
  current-query mean of shipped alignment-head cross-attention under (x,cB,y_<t);
  mask to valid heard frames, renormalize, earliest argmax; 1.0 s centered
  waveform window shifted within seen audio; no DTW/future tokens.
LOCAL_SUPPORT:
  l_t=log((pi_t(E)+1e-12)/(pi_t(M)+1e-12));
  l_null=same native LID odds on all-zero 30 s waveform;
  E_t=max(0,tanh((l_t-l_null)/2)).
BASELINE_CONFLICT:
  P_M=sum_{v in V_M} pB_t(v); P_E=sum_{v in V_E} pB_t(v);
  Q=P_M+P_E (diagnostic); R_t=max(0,P_M-P_E).
ELIGIBILITY:
  incomplete UTF-8 content prefix, localizer/provider failure, or nonfinite
  inputs -> g_t=0 with logged reason; no low-Q cutoff; EOS/gap slots retained.
GATE:
  g_t=E_t*R_t on eligible steps; ranking/strength score, not error probability.
BASELINE:
  cB=cM=forced zh ([50258,50260,50360,50364]); cE is forced en
  ([50258,50259,50360,50364]); known M/E supplied as task metadata.
MANDATORY COMPARATOR:
  B0_AUTO, ordinary unsteered Whisper auto-LID on matched IDs.
PRIMARY MECHANISM TARGET:
  English matrix-language confusion: wrong-language substitution and
  phonetic transliteration/script conversion.
SECONDARY:
  EN-deletion-slot, conditional on causal window and conflict at the gap.
OUT OF MECHANISM SCOPE:
  same-language English substitution; retain in overall PIER/WER.
```

**Exact next action:** Freeze and implement the P0-R2 feasibility specification from this reconciled contract.
