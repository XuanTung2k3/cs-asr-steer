# P0-R2 repairability feasibility specification — v1.1 (v1 + pre-run implementation completion, §7)

**Status:** frozen scientific definition and implementation contract for the next *unsteered* R2 feasibility run. This document supersedes only the **primary localizer and primary gate** in `P0_R2_FORMULA_RECONCILIATION.md`. That reconciliation remains the source for LS-B, BC-B, known-M baseline, scope, and data-role boundaries. The independent design audit and historical P0/P0-R1 specs/reports remain unchanged. No R2 outcome has been observed and P1 has not started.

## 1. Scientific change and causal claim boundary

The old `g_old=E R_B` asks whether local English evidence conflicts with forced-ZH baseline script mass. The proposed primary `g_cf=E [R_B-R_Ecf]_+` additionally asks whether **the same-prefix forced-English condition reduces that conflict**. It is a repair-need ranking/strength score, not a calibrated error probability or proof that steering will correct a word. The Ecf branch changes only `<|zh|>` to `<|en|>`; it never generates its own transcript.

The primary mechanism target stays `wrong_language_substitution` plus `phonetic_transliteration_or_script` for embedded English under Mandarin matrix-language confusion. `EN-deletion-slot` is secondary. `EN-same-language-sub` is out of the mechanism but remains in overall PIER/WER. No general English-error or general multilingual claim follows from this script partition.

For this known-M experiment, M and E are supplied task metadata. `B0` is greedy forced ZH, `cB=cM=[50258,50260,50360,50364]`, and `cE=[50258,50259,50360,50364]`; `hB=hM` remains the expected P0 identity. `B0_AUTO`, ordinary unsteered auto-LID Whisper on the same IDs, is a mandatory comparator. Both recognition baselines must be reported; no proposed-method claim may hide B0_AUTO.

## 2. Exact inference-time computations

All signals at token position `t` use full audio `x` truncated by Whisper to its heard 30 s, the actual B0-generated prefix `y^B_<t`, and the current query predicting `y^B_t` (or EOS slot). The R2 runner uses frozen Whisper-large-v3, bf16, eager attention, eval mode, greedy beam 1, temperature zero, max 200 new tokens, task transcribe, no timestamp generation, and no previous-token conditioning. A full teacher-forced replay of the fixed B0 content IDs with `use_cache=False` is allowed because decoder self-attention is causal. Query index is `len(prompt)+t-1`, including the final prompt query at `t=0`. The Ecf replay receives the **same content IDs**, never a separately generated English continuation. Its prompt differs at exactly the language-token index. Both branches use the same real-audio encoder output and precision. No reference, evaluator label, CTC timing, or future generated token enters either next-token score.

### 2.1 Causal acoustic window

From the shipped `generation_config.alignment_heads`, take each head's softmax cross-attention at the current B0 query. Average the frozen heads, mask all frames beyond `F=min(1500,ceil(heard_samples/320))`, and renormalize to unit sum. Validity requires a nonempty, finite, nonnegative vector with positive valid-frame mass. For every contiguous 50-frame interval `[s,s+50)` within `F`, sum its normalized mass and select the **earliest maximum**. If `F<50`, take `[0,F)`. This is the only primary window rule. Its raw PCM crop has 16,000 samples at 16 kHz if heard audio is at least 1 s; near a partial terminal frame, shift the crop left by at most 319 samples to stay in the heard waveform. For audio shorter than 1 s, use all heard samples. Save start/end frame, start/end sample, actual duration, selected attention mass, and the old earliest point-argmax frame. The old centered 1 s point window is a timing ablation only; it cannot replace the primary after seeing gate outcomes. Neither rule uses post-hoc DTW, cross-query normalization, future tokens, or CTC time.

### 2.2 LocalSupport LS-B — unchanged

Apply Whisper's native language-token readout to the **actual waveform crop** padded by the feature extractor to 30 s. Use only the single `<|startoftranscript|>` decoder query and softmax over its 100 language-token logits. The all-zero 30 s waveform provides a frozen null odds under the same model, preprocessing and precision. With `eps=1e-12`:

```text
l_t    = log((pi_t(en)+eps)/(pi_t(zh)+eps))
l_null = log((pi_null(en)+eps)/(pi_null(zh)+eps))
A_t    = l_t-l_null
E_t    = max(0,tanh(A_t/2))
```

Log `pi_t(en)`, `pi_t(zh)`, pair mass, signed `A_t`, `l_t`, `l_null`, and `sigmoid(A_t)`; only `E_t` enters the gate. `A=0` gives exactly `E=0`. This is null-relative model evidence, neither a calibrated English probability nor a physical acoustic likelihood ratio.

### 2.3 Baseline and Ecf conflicts — same frozen partition

The tokenizer partition is `whisper_han_ascii_v1`: `V_M` is pure Han (optionally CJK punctuation) plus incomplete Han-lead UTF-8 fragments; `V_E` is ASCII-Latin word tokens, including apostrophe-leading contractions; all other IDs are ambiguous. It is disjoint and exhaustive, frozen by `sha256:7daa50091056677286dd30933300cb9b1245ca778f969df2a95ccc06016dac02` in the config and manifest. The expected local tokenizer counts are `1667/37858/12341` for M/E/ambiguous. These groups measure **script affinity**, not lexical language. Mid-character content prefixes are ineligible and detected from raw tokenizer bytes under strict UTF-8, never replacement text. Log the ambiguous mass and conditional margin, but do not use either to veto a step.

For branch `j ∈ {B,Ecf}`, upcast its full-vocabulary logits to float32 **before** `log_softmax` and compute:

```text
P_M^j = sum_{v in V_M} p_t^j(v)
P_E^j = sum_{v in V_E} p_t^j(v)
Q^j   = P_M^j+P_E^j
R^j   = max(0,P_M^j-P_E^j)
```

Validate separately that `0≤P_M^j,P_E^j≤1`, `0≤R^j≤Q^j≤1` within `1e-6`, with finite logits/masses. There is no low-Q abstention.

### 2.4 Repairability and two predeclared gates

```text
D_t    = max(0,R_t^B-R_t^Ecf)
g_cf_t = E_t D_t                         # new primary
g_old_t= E_t R_t^B                       # frozen old-gate ablation
```

Require `0≤D≤R_B≤Q_B` and `0≤g_cf≤g_old` per position. Do not divide D by R_B, multiply by R_B again, or threshold E/D/Q. The pointwise upper bound means the new gate cannot increase Mandarin activation at any fixed absolute threshold; **better ranking remains an empirical question**.

### 2.5 Structural fallback and audio control

Use this reason precedence: `mid_character`, `localizer_fail`, `local_support_fail`, `baseline_provider_fail`, `ecf_provider_fail`, `nonfinite_signal`. A failure of anything `g_old` needs (prefix, localizer, LocalSupport, baseline) sets `D=g_old=g_cf=0` (`eligibility_status=ineligible`). A failure confined to the Ecf branch sets `D=g_cf=0` with reason `ecf_provider_fail` but **preserves `g_old=E·R_B`** (`eligibility_status=cf_ineligible`), because every input of the old gate is valid. Every fallback remains in the dense evaluator panel. No low-Q, low-D, low-E, confidence, or dynamic-window veto exists. EOS is not hard-vetoed: an end deletion can map to that slot if B0 emitted EOS. The installed `generate` strips a final EOS from its returned sequence, so "B0 emitted EOS" is defined as `EOS ∈ sequence` **or** content length below `max_new_tokens` (short-form greedy stops only on EOS or the cap). If B0 stops only at the 200-token cap, the terminal gap is explicitly unalignable.

For the deterministic cyclic-permutation control, keep the B0/Ecf full-audio logits, prefixes, `R_B`, and `R_Ecf` **fixed**. Replace only the selected local waveform by the donor waveform at the same seconds; if the donor ends earlier, shift the crop left, and if it is shorter than 1 s use all available donor audio. Recompute only `pi`, `A`, and `E`, then both control gate values using the fixed conflicts. A control provider failure is logged and does not invalidate the real-audio gate. This tests LocalSupport's response to matched audio; it does not test whether the Ecf conflict is acoustically causal.

## 3. Data, alignment, and stored artifacts

The immutable population is the **same 300 utterance IDs/20 dialogues** in DG-04 B0's `D-dev-select` result. Build a new dense R2 panel, sorted by utterance ID, with every generated content query and an EOS slot only when generated. Never reuse P0 labels or coarse positions. The inference panel contains only audio identity/path/duration and donor ID. A separate evaluator panel contains reference/dialogue data. The R2 inference runner cannot read the evaluator panel.

After the run, compute canonical PIER reference↔B0-hypothesis unit alignment on the **fresh exact R2 forced-ZH baseline**. Map hypothesis units to their first generated token query through checked token/normalization offsets. A deletion maps to the next aligned hypothesis unit's query or emitted EOS; consecutive deletions sharing a slot count as **one position** in AUROC. Missing/ambiguous offsets, stratum collisions, boundary cases, beyond-heard-audio targets and capped terminal gaps remain `unalignable:*`, counted by reason. Evaluator-only MMS-FA/CTC unit midpoints validate both localizers, never supply their windows. Keep same-language English substitutions in corpus PIER/WER and in a separate diagnostic stratum.

Every run manifest records resolved config/config hash, Git commit and clean state, environment, model/tokenizer/partition revisions, source and data hashes, conditions, population hashes, seed, and status. Each atomically written utterance shard records raw B0 IDs/text, prompt and shared prefix IDs, query index, full branch masses, LS-B components, null odds, both gate values, both windows' indices, donor crop, fallback and timing. CPU aggregation independently recomputes row invariants, stores category/offset/timing records, and reports B0_AUTO on matched IDs. Shards are resumed only when identity and manifest hash match. Nothing writes to P0/P0-R1 result directories.

The runner verifies each source and donor audio against the manifest's size-plus-first-65,536-byte hash before decoding and aborts before any row if the frozen CUDA/bfloat16 execution setting is unavailable. The evaluator records every structural fallback and `unalignable:*` reason; position collisions make all affected reference units unalignable for the primary contrasts.

## 4. Pre-run feasibility and preference rules

Thresholds are operational feasibility bars, not calibrated probabilities. Use **2,000 dialogue-cluster bootstrap resamples**, seed `240924`, percentile 95% intervals for **both AUROC and AUPRC**. Each primary AUROC comparison needs at least **30 unique mapped positions and 10 dialogues in each stratum**; otherwise `NOT_ESTIMABLE`. For gate AUROC, include mapped structural fallback positions with gate zero. Report AUROC, AUPRC, intervals, positions, dialogues, per-stratum median/IQR/5th–95th quantiles, and fallback counts. The full numeric config is `configs/inference_cf/p0_r2_repairability.json`.

| Check | Frozen pass rule |
|---|---|
| ALIGN | ≥80% of non-boundary English reference POIs map to a unique token query/gap slot; report all reasons. |
| LOCALIZER | ≥60% of timed mapped units have a valid maximum-mass window center within 0.5 s of evaluator CTC midpoint; failures count as misses; at least 30 timed English and 30 timed Mandarin units across ≥10 dialogues each. Report both localizers' valid rates, center errors, within-0.5 s rates, actual-window midpoint coverage by `ZH-correct`, `EN-correct`, `EN-confusion`, deletion slots. No outcome-based localizer switch. |
| R2-LS | Finite E on ≥90% of complete-prefix/valid-audio pre-provider steps; AUROC(E; `EN-correct∪EN-confusion` vs `ZH-correct`) ≥0.70 with lower 95% bound >0.5. Matched audio-control coverage ≥90% of finite mapped LS positions; lower bound for paired `AUROC_real−AUROC_mismatch` >0. Null identity `A=E=0` within `1e-6`. |
| R2-BC/Ecf | Every complete-prefix/valid-audio step has finite B and Ecf probability masses and zero partition/formula/bound violations beyond `1e-6`. Report `P_M/P_E/Q/R` by stratum, including EOS and gaps. |
| R2-GATE | `g_cf` AUROC ≥0.70 for **both** `EN-confusion` vs `ZH-correct` and `EN-confusion` vs `EN-correct`, each lower 95% bound >0.5. Report both old/new AUPRC and paired ΔAUROC intervals; do not demand an arbitrary fixed improvement margin. |

`R2_CF_FEASIBLE` requires all five checks, full 300-utterance shard accounting, B0_AUTO comparator coverage, and no leakage/provenance violation. `R2_CF_PREFERRED` additionally requires a **positive lower 95% paired ΔAUROC bound** for confusion versus correct English and a nonnegative point ΔAUROC for confusion versus Mandarin. Pointwise `g_cf≤g_old` supplies the no-increase-in-Mandarin-activation guarantee at fixed score thresholds. If feasible but improvement evidence is absent, report `R2_CF_FEASIBLE_NOT_PREFERRED`; do not select the new gate merely by name. Any failed component gives `R2_BLOCKED`. Neither formula nor criterion may be revised from R2 outcomes in this run. R2 does not authorize P1 automatically.

## 5. Adversarial pre-execution self-audit

1. **Can D be large from a trivial forced-English script shift?** Yes. Ecf changes a prompt token, and D can be high on Mandarin or even silence. D alone is not evidence of English speech; report its full distribution on `ZH-correct`, `EN-correct`, confusion, and gap slots.
2. **Does E sufficiently prevent that?** It guarantees zero only at or below the measured null odds. Whether it suppresses Mandarin speech, adjacent English bleed, and noisy silence is **unproven** and is tested by LS discrimination, matched-audio control and Mandarin gate diagnostics.
3. **Can evaluator information enter inference?** The inference API accepts only audio, token IDs/prefix, prompts, frozen tokenizer classes and model scores. References/dialogues/CTC exist in a separate panel read only by the CPU evaluator. Code and tests check this boundary.
4. **Are branch prefixes identical?** The prompt differs only at language-token index 1; content IDs are exactly B0 `content[:t]` for both. Tests check this and query indices. No Ecf continuation is generated.
5. **Can the window see unheard frames?** Valid frames are capped by actual heard samples and 1,500 frames. The chosen sample crop is bounded by the heard waveform. Tests cover padded-tail peaks and partial terminal frames.
6. **Are thresholds outcome-driven?** All numbers above and in the versioned config are fixed before R2 output. No R2 gate outcome has been inspected. The older 0.70 feasibility floor is retained; no new favorable AUROC margin is invented.
7. **Is the partition pair-specific?** Yes. Han versus ASCII-Latin is usable only for this Mandarin–English/script setting. Same-script transfer requires a separately frozen provider.
8. **Does the contract claim general LID?** No. Whisper short-window LID is used as null-relative model evidence; its local validity is exactly the R2 question.
9. **Could high `g_cf` mean prompt susceptibility rather than repair need?** Yes. A positive Ecf effect is necessary for this proposed score but does not prove steering improves recognition. High scores on Mandarin, names, or incorrect localization would undermine the interpretation. R2 is predictive feasibility, not a causal repair result.
10. **What controls distinguish the explanations?** `g_old` isolates the added Ecf term; `D/R_Ecf` by stratum exposes blanket prompt effects; point-argmax timing tests localization rationale; matched versus cyclic mismatched local audio tests E's acoustic dependence; B0_AUTO tests a stronger non-steered decode policy. A future outcome-blind full-audio Ecf shuffle/zero-audio diagnostic could test D's audio dependence, but is **not** introduced as an R2 selection lever. Only a later authorized P1 intervention can test actual repair.

**Pre-execution verdict:** no fatal algebraic or inference-leakage flaw is known. The key unresolved scientific risk is that Ecf conflict reduction may be prompt susceptibility, while LS-B may fail to veto it on Mandarin speech. This is a measured R2 feasibility risk, not grounds to alter the formulas after outcomes.

## 6. Execution boundary

R2 may run only after this spec, config, and code are committed and the clean-state manifest is generated. One physical GPU job is the intended feasibility run; rerun only a concrete invalid implementation/infrastructure failure, never to choose a formula or window from outcomes. **No activation steering, direction injection, alpha optimization, training, P1 experiment, D-dev-confirm selection, or D-test use is permitted in R2.**

## 7. v1.1 — pre-run implementation completion (2026-09-24, before any R2 GPU outcome)

An interrupted implementation session was recovered and completed. **No formula, threshold, window
rule, tokenizer partition, population or data role changed.** No physical R2 outcome existed or
was inspected; the only data examined were already-exposed D-dev-select baseline transcripts,
references and evaluator-only CTC times (no gate or LocalSupport value), plus a synthetic
random-gate evaluator exercise.

| Item | v1 implementation | v1.1 | Reason |
|---|---|---|---|
| EOS gap slot | created only if `EOS ∈ sequence` | `EOS ∈ sequence` or content below cap | `generate` strips the final EOS (`generation_whisper.py:1096`); a CPU replay of two utterances showed the cB replay argmax equals the generated tokens **and** the stripped EOS (13/13, 12/12). v1 would have created no EOS slot and mislabelled end deletions `truncated_gap`. |
| Fallback scope | every fallback zeroes both gates | Ecf-only failure keeps `g_old=E·R_B` | `g_old` does not depend on the Ecf branch; the user-approved addendum requires preserving it. `local_support_fail` is now an explicit reason. |
| ALIGN accounting | stratum-collision units counted as unmapped | a unit is **mapped** if it has a uniquely determined query or gap slot; collisions are excluded only from primary contrasts | Reconciliation §5/§6 separate the two ("map to a unique token position or gap slot" vs "a position with conflicting stratum labels is excluded from primary AUROC"). A deletion's gap slot *is* the next unit's query, so the v1 accounting made ALIGN fail structurally (evaluator dry-run on DG-04 B0 text: 0.797 contrast-eligible vs 0.962 mapped). Both numbers are reported; the frozen 0.80 floor is unchanged. |
| Deletion slot | next aligned unit with unmapped offset fell through to the EOS slot | `unalignable:offset` | Never force a label onto a convenient position. |
| Statuses | pass/fail booleans | `PASS` / `FAIL` / `NOT_ESTIMABLE` per check; `NOT_ESTIMABLE` blocks | Under-powered contrasts are never passes. |
| R2-ECF | implicit | explicit check: every row `same_prefix_validation`, finite Ecf masses on complete-prefix steps, zero invariant violations | Addendum §13. |
| Invariants | checked on eligible rows only | checked on every computed branch, including fallback rows; violations are recorded and block BC/ECF | Zero-violation rule applies to every step. |

**Diagnostic-only additions (never a gate, never a pass rule, never a selection lever):**

- **Wrong-language counterfactual `cX` = forced Russian** `[50258,<|ru|>,50360,50364]`, predeclared here before any outcome: `R_X` on the same prefix, `D_X=[R_B−R_X]_+`, `g_X=E·D_X`. Russian is non-Han, non-Latin and unrelated to both languages, and its script falls in the ambiguous class. `D_X≈D` would mean the conflict reduction is generic "leave Mandarin" prompt susceptibility rather than English-specific. It costs one extra full replay per utterance.
- **Permutation control:** shuffle `E` or `D` among eligible rows (seed 240924) and recompute `E·D`. Discrimination should collapse.
- **Component decomposition:** AUROC of `E`, `R_B`, `R_Ecf`, `D`, `D_X` separately for both primary contrasts.
- **Replay consistency:** per-row `replay_argmax_matches_baseline` and per-utterance agreement. Conflicts are defined on **raw model logits** (before generation logits processors such as `suppress_tokens`); the agreement rate documents how closely replay reproduces the actual decode.

**Engineering additions:**

- **Exact-window LID cache:** an identical crop of identical audio is an identical model input; nothing else is reused.
- **Row provenance:** `baseline_condition_ids`, `english_condition_ids`, `replay_mode`, `same_prefix_validation`, `localizer_status`, `eligibility_status`, window seconds.
- **Shard provenance:** git commit/branch, config hash, model/tokenizer revisions, partition hash, seed.
- **Manifest sources** now also hash `core.py`, the Whisper loader, hashing, normalizer and the MER/PIER evaluators, and the audits.

**Term dependencies (mismatched-audio control).**

| Term | Depends on | Mismatched-audio control |
|---|---|---|
| `E_t` | the selected local waveform crop | recomputed from the donor crop |
| `R_B` | the forced-ZH decoder distribution on the full heard audio | held fixed |
| `R_Ecf` | the forced-EN decoder distribution on the same full heard audio and prefix | held fixed |

The control therefore tests only LocalSupport's dependence on local speech, not the acoustic causality of `D`. This is documented, not a hybrid claim. Donors are the next utterance ID and are often the same speaker/session, which holds speaker and channel roughly fixed while changing content.

**Terminology and claim boundary.** "Repairability" in file and field names is a label for
`D_t`, the **counterfactual conflict reduction**. Before P1 the only permitted descriptions are
*counterfactually language-responsive matrix(-script) conflict* and *inference-time repair-need
ranking/strength score*. R2 cannot establish proven repairability, a probability that
intervention will fix a token, a valid steering direction, or any MER/PIER gain.

**Contrast interpretation.**

- **EN-confusion vs ZH-correct:** both strata emit Han, so success there needs acoustic
  evidence. Success on this contrast alone could still be English/Mandarin LID.
- **EN-confusion vs EN-correct:** mostly separated by `R_B` because of the emitted script (Han vs
  Latin). It confirms the gate stays low on correctly handled English, but it is weak evidence of
  failure detection beyond "English audio, Han emitted".

Both must pass, and reports must show the component decomposition.

**Plan provenance.** The user-approved addendum names
`INFERENCE_STEERING_IMPLEMENTATION_PLAN_REPAIRABILITY.md`; no such file exists on this machine.
Its method content (localizer, LS-B, BC-B, Ecf, D, both gates, fallbacks, criteria) was taken
from the addendum text itself and matches this spec; `INFERENCE_STEERING_IMPLEMENTATION_PLAN.md`
is the committed plan.
