# P0-R2 repairability — independent adversarial audit (pre-run)

**INDEPENDENT AUDIT VERDICT: PASS_TO_R2_RUN**

**Scope and evidence.**

- **Audited artifacts:** the recovered and completed P0-R2 implementation on
  `feature/inference-cf-steering`, against `P0_R2_FORMULA_RECONCILIATION.md`,
  `P0_R2_REPAIRABILITY_SPEC.md` v1.1 and the user-approved repairability addendum.
- **Plan file:** the addendum names `INFERENCE_STEERING_IMPLEMENTATION_PLAN_REPAIRABILITY.md`,
  which does not exist on the machine; its method text was audited directly.
- **Data examined:** no physical R2 outcome exists (`results/inference_cf/p0_r2/` is absent) and
  none was inspected. The only data examined were:
  - already-exposed D-dev-select baseline transcripts, references and evaluator-only CTC times;
  - a CPU replay of two utterances (baseline tokens only);
  - a synthetic random-gate evaluator exercise.
- **No P1 intervention exists or ran.**

## 1. Recovery summary

| Source | Content |
|---|---|
| Interrupted Codex session (uncommitted) | spec v1; config; `core_r2.py` algebra; 50-frame max-mass localizer; full same-prefix B/Ecf replay runner; donor control; clean-tree prepare; evaluator; 16 tests |
| Completed in recovery | EOS-slot rule; Ecf-only fallback scope preserving `g_old`; explicit `local_support_fail`; ALIGN mapping accounting; deletion offset fallthrough; `NOT_ESTIMABLE` statuses; explicit R2-ECF check; branch invariants on every row; diagnostic wrong-language (`ru`) counterfactual, permutation control and component decomposition; replay-agreement diagnostic; exact-window LID cache; row/shard provenance; 12 more tests; spec v1.1; docs |

## A. Mathematical fidelity — PASS

The code implements exactly these formulas (`core_r2.py`):

- `E=[tanh(A/2)]_+`, with `A=log((π_E+ε)/(π_M+ε))−log((π∅_E+ε)/(π∅_M+ε))` and ε=1e-12.
- `R=[P_M−P_E]_+` for both branches, via one function (`conflict_from_logits`).
- `D=[R_B−R_Ecf]_+`, `g_old=E·R_B`, `g_cf=E·D`.

Checks against accidental changes:

- **No normalized margin in any gate.** The conditional margin exists only as a logged
  diagnostic.
- **No division by or second multiplication with `R_B`.** No threshold appears in the gate path
  (searched).
- **ε is used only inside logs.**
- **Numerics:** logits are upcast to float32 before `log_softmax`; masses are
  `exp(logsumexp)` in float32, so there is no bf16/float16 underflow in the masses.
- **Partition:** `whisper_han_ascii_v1` is disjoint and exhaustive (1667/37858/12341). Its hash
  `7daa5009…` is identical in config, prepare and tests.
- **Bounds:**
  - `0≤D≤R_B≤Q_B` and `g_cf≤g_old` are asserted per row.
  - `E<1` holds numerically: the maximum attainable `A≈24.89` gives `E=1−3.1e-11`.
- **Tests:** null identity `A=E=0`, monotone LS-B for `A>0`, and the float32 computation are all
  covered.

## B. Same-prefix counterfactual and KV cache — PASS (highest priority)

`inference_utterance` performs one decoder forward per branch with `use_cache=False`, an
all-ones mask and implicit positions `0..L−1`. Across branches:

- **Same inputs:** the same encoder output object (same heard audio, truncated to 30 s), the same
  content IDs (the actual B0 `content[:t]` list) and the same query index `len(prompt)+t−1`.
- **Prompts:** `same_prefix_inputs` raises unless the prompts differ only at the language index
  (`<|zh|>` 50260 → `<|en|>` 50259); `<|transcribe|>` and `<|notimestamps|>` are identical.
- **No generation:** no Ecf token is generated or fed back.
- **Per-row proof:** each row records `baseline_condition_ids`, `english_condition_ids`,
  `replay_mode=full_same_prefix_replay_use_cache_false` and `same_prefix_validation`.

**Cache hot swap is invalid.** A tiny-Whisper test shows that swapping the language token changes
every later decoder state, so KV entries built under cB cannot serve cE. The runner never swaps.

**Future tokens.** Replaying the fixed B0 sequence is equivalent to prefix-only computation:
another tiny-Whisper test gives the same logits and cross-attention at the earlier query.

**Replay reproduces generation.** A CPU check with the real model showed the cB replay argmax
equals the generated tokens and the stripped final EOS on 2/2 utterances (13/13, 12/12). The run
logs agreement per row.

**Caveat (documented).** Conflicts are defined on raw model logits, before `suppress_tokens` and
the other generation logits processors.

## C. Causal localizer — PASS

- **Inputs:** only the current query's attention from the 10 shipped alignment heads.
- **Heard frames only:** frames beyond `F=min(1500, ceil(heard_samples/320))` are cut before
  renormalization. Tests confirm that a padded-tail attention peak cannot move the window.
- **Window selection:** exact 50-frame (1.0 s) window; earliest maximum via `np.argmax` (ties
  among bitwise-equal float64 sums); crop of 16,000 samples, shifted left by at most 319 samples
  near a partial terminal frame; all heard audio if shorter than 1 s.
- **Point argmax:** diagnostic/ablation only.
- **Absent from inference:** DTW, CTC, reference timing, future query, label-driven shift and any
  2 s fallback (searched and tested).
- **Unproven, measured risk:** prior repository evidence of attention sinks. Localizer superiority
  must be judged by the evaluator timing comparison, never by gate AUROC.

## D. Leakage — PASS

- **Runner / core:** `inference_utterance` accepts only audio paths/hash, prompts, partition,
  null odds and model. The runner reads only `inference_panel.json`, which holds audio
  identity/path/duration and donor ID.
- **Evaluator-only data:** references, dialogue IDs, CTC times and the B0_AUTO comparator live in
  `evaluation_panel.json` and parquet files that only the CPU evaluator reads. The evaluator never
  imports the runner.
- **Evidence:** a signature test and a code search confirm this separation.
- **Population:** fixed by DG-04 B0 IDs, independent of gates.

## E. Prompt susceptibility — RISK, correctly scoped

`D` can be large on ordinary Mandarin if forced `<|en|>` mechanically moves mass from Han to Latin
(or toward translation-like output). In that case `D≈R_B` and `g_cf≈g_old`: a prompt-susceptibility
score, not repair need.

**Guarantees.** `g_cf≤g_old` pointwise, so the Ecf term can never *increase* Mandarin activation.

**Grounding.** `E` is the only acoustic factor, and it suppresses Mandarin only if short-window
LocalSupport actually separates English from Mandarin speech. That is the R2-LS question and is
unproven.

**Diagnostics that expose it:**

- `D`, `R_Ecf` by stratum;
- component AUROCs of `E` and `D`;
- the diagnostic wrong-language `D_X`/`g_X` with forced Russian: `D_X≈D` implies generic
  "leave-Mandarin" susceptibility;
- permutation controls;
- the paired ΔAUROC.

**Consequence.** If the Ecf term adds nothing, the frozen outcome is
`R2_CF_FEASIBLE_NOT_PREFERRED` (or blocked). No formula may change.

**Terminology.** "Repairability" is only a label for `D` = counterfactual conflict reduction.
Before P1 the permitted wording is *counterfactually language-responsive matrix(-script) conflict*
and *repair-need ranking/strength score*.

## F. Repair need vs English detection — RISK, correctly scoped

The two primary contrasts carry different information:

| Contrast | What it tests |
|---|---|
| **EN-confusion vs ZH-correct** | Both strata emit Han, so success needs acoustic evidence. It could still be English/Mandarin LID restricted to Han-emitting positions. |
| **EN-confusion vs EN-correct** | Mostly separated by `R_B` through the emitted script (Latin ⇒ `R_B≈0`). Near-tautological: it shows the gate stays quiet on correctly handled English, not that it detects errors that remain Latin. |

**Requirements.**

- Both contrasts must pass, and reports must show the component decomposition.
- The gate may be called a detector of *English speech rendered in the matrix script*, never a
  generic English-error detector.
- Same-language substitutions are out of scope by construction but stay in PIER/WER.

## G. Script specificity — PASS (scoped)

`V_M`/`V_E` measure Han vs ASCII-Latin script affinity; pinyin and other Latin-script words count
as "E". R2 supports only the distinct-script Mandarin–English feasibility claim. The docs contain
no same-script, multilingual, LID-general or architecture-independent claim.

## H. Statistics and evaluator — PASS (after recovery fixes)

- **Panel:** dense over all generated queries plus EOS slots.
- **Labels:**
  - labels come from the fresh forced-ZH R2 baseline; `B0_AUTO` is used only for comparator
    metrics, never labels;
  - canonical PIER alignment is used;
  - hypothesis units map to their first-token query (0 map failures on the 57 real P0 ID
    sequences and on 300 B0 texts);
  - deletion gap slot = the next aligned unit's query or EOS;
  - collisions and unmapped offsets are explicit `unalignable:*`, with collision exclusions
    reported per stratum (dry-run: 72 ZH-correct, 26 EN-correct, 19 EN-confusion, 276
    deletion units).
- **Retention and power:**
  - fallback rows stay at `g=0` in gate AUROC;
  - dialogue-cluster percentile bootstrap: 2000 resamples, seed 240924, valid-replicate counts
    reported;
  - minimum 30 positions and 10 dialogues per compared stratum, otherwise `NOT_ESTIMABLE`, which
    blocks.
- **ALIGN accounting** (recovery change): mapping vs contrast eligibility, justified from the
  reconciliation text and disclosed with both numbers. The floor is unchanged.
- **Synthetic check:** a synthetic random-gate run gives AUROC≈0.51, so there is no label leakage
  in the evaluator.

## I. B0_AUTO visibility — PASS

- **Summary:** reports PIER (with categories), MER, English WER and Mandarin CER for B0
  forced-ZH and B0_AUTO on matched IDs. Missing matched IDs block the run.
- **Reproduced development gap:** B0 PIER 0.470 vs B0_AUTO 0.390.
- **Rule:** no result beating B0 alone may be called superior to ordinary Whisper.

## J. Reproducibility — PASS

- **Manifest:** git commit and branch (clean tree required), resolved config and hash,
  model/tokenizer revisions, partition hash and counts, prompts incl. diagnostic `cX`, decode,
  seed, environment, population/panel hashes, and source hashes of spec, reconciliation, audits,
  plan, code, evaluators and data.
- **Shards:** provenance, raw B0 IDs and text, prefix IDs, query index, window
  frames/samples/seconds, all components, both gates, fallback and eligibility, control, LID call
  counts.
- **Resume:** only on identity plus manifest-hash match.

## Claim boundary

| Status | Claims |
|---|---|
| **Supported if R2 passes** | Causal local audio windows carry usable excess English evidence. Baseline matrix-script conflict is measurable. Same-prefix English conditioning reduces that conflict at relevant positions. `g_cf` discriminates the specified EN-confusion positions under the frozen criteria. |
| **Not supported until P1** | Activation steering repairs ASR. `hE−hB` is a valid beneficial steering direction. A high `g` guarantees a beneficial intervention. Steering improves MER/PIER. |
| **Not supported by this R2 setup** | Same-script language generalization. Universal multilingual repair. Architecture independence. General language identification. |

## Findings

**CRITICAL:** none remaining. The pre-recovery EOS-slot defect would have silently removed every
end-of-utterance deletion slot; it is fixed and tested.

**MAJOR (fixed):**

1. ALIGN conflated label collisions with mapping failures, so it would fail structurally.
2. A deletion with an unmapped next offset was forced onto the EOS slot.
3. An Ecf-only failure zeroed `g_old`, contrary to the approved contract.
4. Under-powered contrasts were reported as plain fails rather than `NOT_ESTIMABLE`.

**MAJOR (measured scientific risks, not blockers):**

1. Prompt susceptibility of `D` (E).
2. Contrast 2 near-tautological through `R_B` (F).
3. Short-window LocalSupport validity and attention-sink localization are unproven.
4. Collisions leave the secondary deletion stratum small.

**MINOR:**

1. Conflicts use raw logits (pre-`suppress_tokens`).
2. Earliest tie-breaking is exact only for bitwise-equal float sums.
3. Donors are often the same speaker/session (content control, not speaker control).
4. A failure shard is not auto-retried on resume and must be removed deliberately with a recorded
   reason.
5. The named repairability plan file is absent.

## Verdict checklist

| Requirement | Status |
|---|---|
| implementation complete | ✅ |
| formulas exact | ✅ |
| same-prefix Ecf valid | ✅ |
| KV-cache/replay valid | ✅ |
| causal localizer valid | ✅ |
| no inference leakage | ✅ |
| old gate preserved | ✅ |
| new gate implemented | ✅ |
| fallback accounting correct | ✅ |
| criteria frozen | ✅ |
| focused tests passing | ✅ (45 inference-CF tests) |
| B0_AUTO preserved | ✅ |
| claims scoped | ✅ |
| P1 not executed | ✅ |

**INDEPENDENT AUDIT VERDICT: PASS_TO_R2_RUN**

**Next action:** commit, prepare the clean-state manifest, then run the frozen P0-R2 feasibility
experiment only (one physical job). Do not start P1 steering.
