# CS-ASR-ReFT P0/P1 protocol — frozen subspace feasibility

**Status: FROZEN** before any P0 decode, before any P1 decode or gradient capture, and before any reference access.
- Freeze records are written once under `/mnt/data/tungnx/cs-asr-reft/freeze/` (`p0.json`, `p1.json`). Each holds
  source hashes and the git head.
- Constants are in `configs/reft/p0p1.yaml`.
- Amendments are appended in §12, timestamped and justified, and only before the outcome they could affect.

## 1. Research questions

- **P0.** Does the unmodified Whisper-large-v3 Auto baseline (beam 5) still make enough genuine code-switching lexical
  errors to justify adaptation?
- **P1.** Can task-informed correction gradients pick out a rank-4 frozen subspace at one fixed decoder site that has
  useful local lexical-correction leverage beyond random directions, on held-out training-audit examples?

P1 measures three things:
1. representation/gradient separability;
2. gradient energy retained by the subspace;
3. local oracle controllability.

P1 does not measure free-decoding ASR improvement.

**Later intervention (not implemented or trained here).** The intervention the sprint is testing for is:

h̃ = h + α·V·tanh(A·Vᵀ·Norm(h) + b)

Here V ∈ ℝ^{1280×4} is frozen and A, b are learned in P2.

## 2. Model and data roles

**Model.**
- Whisper-large-v3 at `/mnt/data/tungnx/whisper-large-v3`, in BF16 with eager attention asserted on all 96 attention
  modules.
- Identity check: sha256 of `model.safetensors` must be `a8e94b85…`, and the BF16 parameter hash must be `0b2acef4…`.
  Both are anchored to ACL-0 E0 (Slurm 59059).
- All parameters are frozen (`requires_grad=False`). Weights are hashed before and after every GPU job.

**P0 population.** The ACL-0 dev panel: 150 utterances from 20 `D-dev-select` dialogues.
- Identity is taken from ACL-0 `panels.json` (sha256 `6741eb36…`) and verified against the role manifest by dialogue,
  conversation, speaker and audio sha256.
- It is exposed (A3-ACL0-DEV150, used for ACL-0 checkpoint selection). It serves for exploratory baseline diagnosis only.

**P1 population.** A new panel from `loc-train ∪ util-train`, which contains no dev, test or confirmation dialogue.
- Exclusions:
  - the 512 ACL-0 training utterances;
  - utterances outside 1–30 s;
  - utterances whose target has more than 190 tokens or whose target units do not equal the reference units;
  - utterances with fewer than 3 non-filler units.
- Strata come from reference units. Substantive English means an EN-tagged unit that is not a filler; Mandarin means a
  ZH-tagged unit that is not a filler.
  - CS: both present.
  - EN_ONLY / ZH_ONLY: only one present.
- Quota: CS 400, ZH_ONLY 60, EN_ONLY 40.
- Sampling: within each dialogue, a `default_rng(42)` permutation (IDs sorted), then round-robin over the sorted dialogues.
- Split, dialogue-disjoint: dialogues are taken in `sha256("reft-p1-split:"+dialogue_id)` order into **audit** until the
  audit partition holds ≥ 20% of the panel. All remaining dialogues are **construction**.
- Disjointness from D-dev-select and from the P0 panel is checked on dialogue, conversation, speaker, corpus speaker,
  audio hash and utterance ID. The construction and audit partitions must share no dialogue.

**Never opened:** `D-test`, `D-dev-confirm`, `router-calib`. Their dialogues are known from `assignment.json` only.

## 3. P0 — operational Whisper Auto baseline (F1)

**Decoding.**
- `generate(language=None, task="transcribe", num_beams=5, do_sample=False, temperature=0.0, max_new_tokens=200,
  condition_on_prev_tokens=False)`, batch 1.
- Transformers detects the language as the argmax over the 100 language tokens at `<|sot|>`. The prompt is then
  `[sot, detected, <|transcribe|>, <|notimestamps|>]`.
- No forced `<|zh|>`, no reference-derived language. Same audio pipeline and text convention as ACL-0.
- Logged per utterance: detected language and top-3 language log-probabilities, content tokens, termination, and cap hit
  (200 tokens without EOS).

**Integrity (all required; otherwise `P0_BLOCKED`):**
- data audit valid;
- model identity matches both hashes;
- weights unchanged;
- no trainable parameters;
- engine identity: the same function with `language="zh"` reproduces sealed ACL-0 B0 beam-5 tokens on the first 3 rows;
- API equivalence: `generate(language=None)` equals `generate(language=<detected>)` on 10 rows;
- determinism: 10 repeat decodes are identical;
- audio sha256 matches for all 150;
- the detected language equals an independent argmax for all 150;
- beam 5.

The decode runs behind a reference firewall that refuses to open role manifests, `transcript`, targets and evaluation
files. Outputs are sealed (sha256) before references are read.

**Evaluation.**
- Units: canonical normaliser, one Han character or one Latin word, with the ACL-0 alignment.
- Utterance strata come from reference units (§2): CS, EN_ONLY, ZH_ONLY, OTHER, with denominators reported.
- Systems: AUTO (new), plus sealed ACL-0 beam-5 B0 (forced `<|zh|>`), H1 and A1 at their selected epochs. The H1/A1
  comparison is descriptive and is not a matched adaptation comparison.
- Metrics per system and stratum: MER, English WER, English PIER (wrong EN reference units / EN units), Mandarin CER,
  substantive English errors, non-filler Mandarin errors, non-termination, loops (≥ 15 repeated units), derailment
  (either of the two).
- **AUTO English error kinds** (substantive EN reference units):
  - `translation_like` (Han substitute, including transliteration);
  - `lexical_substitution` (a different English word);
  - `segmentation_artifact` (substitute contains, or is contained in, the reference word);
  - `omission` (deletion, with or without nearby insertions);
  - `boundary` (deleted, but the word appears within ±3 units);
  - `other` (numerals, fillers, other scripts).
- Unit-level B0/H1/A1/AUTO correctness contingency over substantive EN units. This includes "H1 corrected vs B0 and
  AUTO already correct". All four systems are aligned to the same reference units, so no transcript-span alignment is
  invented.
- Further reports:
  - Mandarin errors and new/repaired Mandarin units of AUTO against B0, H1 and A1;
  - utterances improved or worsened;
  - language-ID distribution;
  - concentration (top-1 and top-3 utterance and dialogue shares);
  - dialogue-block bootstrap of ΔMER (2,000 resamples, seed 42).

**P0 target errors (gate quantity).** A target error is a substantive English unit error in a CS utterance whose AUTO
output is not derailed, of kind `translation_like`, `lexical_substitution`, `omission` or `other`.
- Excluded kinds: `segmentation_artifact` and `boundary`.
- A strict count (substitutions only) is reported alongside.

**P0 verdict, in this order:**
1. `P0_BLOCKED` if any integrity check fails.
2. `P0_PASS_RELEVANT_OPPORTUNITIES` if all of the following hold:
   - target errors ≥ 20;
   - target errors span ≥ 5 dialogues;
   - no single utterance holds > 40% of the target errors.
3. `P0_INSUFFICIENT_OPPORTUNITIES` otherwise.

These are feasibility thresholds, not statistical guarantees. If P0 does not pass, P1 is not run.

## 4. P1-A — supervision sources and gradient capture

**Decoding of the P1 panel.**
- F1 Auto beam-5, exactly as in P0 (BF16), behind the reference firewall.
- Training targets (ACL-0 `target_text`: markers and punctuation removed, case kept, no spaces next to Han; `+ EOS`)
  are read only afterwards.

**Local computation precision.** Local computations run on the same weights upcast exactly to float32 (`compute_dtype`)
so that gradients and finite differences are numerically meaningful. Decoding stays BF16.

**Teacher forcing.**
- One forward pass per utterance on `X = [sot, detected, transcribe, notimestamps] + target[:-1]`.
- The prompt uses AUTO's detected language, so the language prefix is compatible with the free decoding.
- The logits at index `3 + t` predict `target[t]`, and the site activation of that decision is at the same index.

**Site.**
- Decoder layer 24 (0-indexed), post-cross-attention residual, pre-FFN: `h = q + s`, with
  `s = encoder_attn(encoder_attn_layer_norm(q), H_enc)`.
- This is the tensor that enters `final_layer_norm` and also forms the FFN residual. It is the same site as DG-02 and
  ACL-0.
- Implementation: a zero tensor `δ` is added to the cross-attention output, so `h → h + δ`. Then `q_i = ∂m_i/∂δ` at
  position `3 + t_i`, computed with `torch.autograd.grad` (no `.grad` accumulation; no parameter has `requires_grad`).
- Not the post-FFN block output. No other layer.

**Margins.** Logit differences equal log-probability differences.
- `m = z(correct) − z(competitor)`.
- Argmax checks use the vocabulary after the generation config's `suppress_tokens`, plus `begin_suppress_tokens` at
  content position 0.

**Row types** (all at matched or gold prefixes; the competitor is chosen without the reference):

- **A — free-decoding first divergence** (primary correction population).
  - The ACL-0 `first_divergence` classifier is applied between the target ids and the AUTO beam-5 content ids.
  - Only `LEXICAL_FIRST/TOKEN` is eligible. That excludes fillers, EOS cases, truncation, `FORMAT_ONLY`, `FORMAT_FIRST`,
    `SEGMENTATION_FIRST` and anomalies.
  - Position `k`; prefix `target[:k]`, which is identical to AUTO's own prefix.
  - Correct token: `target[k]`. Competitor: the token AUTO actually emitted, `base[k]`.
  - An earlier divergence is never skipped.
  - Category from the reference and baseline units:
    - `translation_like` (EN reference, Han baseline);
    - `en_lexical_substitution`;
    - `zh_lexical_substitution`;
    - `zh_to_latin`;
    - `en_omission` / `zh_omission` (the baseline unit is the next reference unit);
    - `insertion`;
    - other.
- **B — teacher-forced gold-prefix errors** (secondary, separately labelled; these are *not* confirmed free-decoding
  failures).
  - Eligible positions `t` (content, EOS excluded) where:
    - the masked argmax ≠ `target[t]`;
    - both tokens are lexical (contain an alphanumeric or non-ASCII byte);
    - the two tokens are not format-equivalent (case, leading space, or prefix relation);
    - the target token's units are non-filler.
  - The A position is excluded. At most 4 per utterance, earliest first.
  - Competitor: the masked argmax, i.e. the highest-probability incorrect token.
- **P — preservation controls.**
  - Matched-prefix positions (`t < k`, or every position when IDENTICAL) whose target token is lexical and non-filler
    and is the masked argmax.
  - At most 4 per utterance, in `sha256("reft-pres:"+id+":"+t)` order.
  - Competitor: the runner-up. Labelled EN or ZH by the token's unit.

**Stored per row** (float32, local only):
- `q`, `h` and `‖h‖`;
- `m₀`;
- correct and competitor ids;
- position, source, category, unit language, utterance stratum, partition, dialogue;
- argmax flags;
- the script-readout gradient `g_s = ∂[LSE z(Latin) − LSE z(Han)]/∂h` at the same position (language-direction
  control, §6).

The Latin set is tokens whose bytes are all ASCII and include a letter. The Han set is tokens containing a Han character
or consisting only of non-ASCII bytes. Special tokens are excluded from both. Gradient norm and orientation statistics
are recorded separately.

**Engineering smoke** (first 4 construction utterances; P1 stops on any failure):
- site identity: the `final_layer_norm` input equals `q + s`, and the δ = 0 logits are bitwise equal to the unhooked
  logits;
- teacher-forced argmax on greedy BF16 tokens matches the cached greedy decode, with mismatches ≤ 2% of steps; this rules
  out a decoder off-by-one, and a 1-shifted alignment must match far worse;
- `q` has shape 1280, is finite and is nonzero;
- the central finite difference along `q̂` equals `‖q‖`, and along a random unit `r` equals `q·r`, within 5% relative
  (step 1e-3·‖h‖);
- `m(h + η q̂) > m₀ > m(h − η q̂)`;
- repeated gradient computation is bitwise identical (no accumulation), and all parameter `.grad` are None;
- weights are unchanged;
- runtime and peak memory are recorded.

The smoke never selects a layer, rank or budget.

## 5. P1-B — weighted SVD and frozen basis

**Basis rows (policy frozen here).**
- If the construction partition has ≥ 100 A rows spanning ≥ 10 dialogues, the basis uses **A only**.
- Otherwise it uses **A ∪ B** (fallback).
- The decision is made on counts only, before any SVD, and never revisited.

**Weights.**
1. Each `q` is normalized to unit norm.
2. Rows of one utterance share weight `1/√n_u`, so each utterance contributes energy 1.
3. If the EN_ONLY or ZH_ONLY utterance stratum exceeds 20% of the total energy, its weights are scaled down uniformly to
   exactly 20%; this is solved jointly. CS rows are never reweighted.

**SVD.**
- `W Q = U Σ Vᵀ` in float64; V is the top-4 right singular vectors, ordered by descending σ.
- Sign convention: `Σ_i w_i (q̂_i·v_j) ≥ 0`, i.e. aligned with the mean correction gradient.

**Checks.**
- `VᵀV = I` (max deviation < 1e-6); V is finite.
- Independent reconstruction: the top-4 eigenvectors of `QᵀW²Q` (`eigh`) span the same subspace (all principal-angle
  cosines > 1 − 1e-6), and the eigenvalues equal σ².

**Reports.**
- the full spectrum (top 64 values and the total);
- explained energy;
- effective rank (exp of the spectral entropy, and the participation ratio);
- in-sample retention per stratum;
- the raw-gradient (unnormalized, unweighted) SVD as a diagnostic, with its principal angles to V.

**Freeze.** V is saved to `p1/basis_rank4.pt` and its sha256 is recorded. V is frozen for the rest of the sprint:
- no optimization;
- no layer search;
- no rank sweep;
- no policy change after evaluation.

**Controls (built at the same time, before evaluation).**
- R1–R5: random orthonormal 4-dimensional subspaces, from QR of Gaussian 1280×4 (float64, `torch.Generator` seeds
  42–46).
- PCA4: the top-4 principal components of the centred `h` of the basis rows (unsupervised structured subspace).
- SCRIPT1: the rank-1 language-preference direction `d = normalize(mean_i ĝ_s,i)` over all construction rows.
- FULL: the unprojected `q̂`, an oracle ceiling and not a control.

## 6. P1-C — held-out training-audit feasibility

Evaluation uses only audit-partition rows; none of them contributed to V.

**Test 1 — retention.**
- `ρ_S(q) = ‖Sᵀq‖²/‖q‖²` for S ∈ {V, R1–R5, PCA4, SCRIPT1}. The random expectation is 4/1280.
- Reports:
  - mean and median;
  - the distribution;
  - per stratum and category;
  - number of examples and dialogues;
  - top-5 example share of the V−R difference;
  - in-sample (construction) against audit retention.

**Test 2 — local leverage (oracle diagnostic).**
- Update: `u_S = ε·‖h‖·P_S q/‖P_S q‖`, applied at the row's site position only, with an otherwise unchanged
  teacher-forced forward pass.
- Budgets: ε ∈ {0.05, **0.10** (primary), 0.25}.
- Reports:
  - `Δm = m_after − m₀`;
  - fraction with Δm > 0;
  - correct beating competitor among rows with m₀ ≤ 0;
  - the correct token becoming the masked argmax;
  - margins before and after.
- It uses the reference-derived `q` only to estimate local controllability. It is never a deployable or reference-free
  method.

**Test 3 — preservation sensitivity** (P rows, predefined perturbations, no oracle selection).
- For each S ∈ {V, R1–R5, PCA4}: `±ε‖h‖·s_j` for each basis vector (8 perturbations). For SCRIPT1: ±d.
- Also the single predefined "mean correction" direction `V·c̄`, where `c̄ = normalize(Σ w_i Vᵀq̂_i)` over the basis
  rows; its random analogue is `R·c̄_R`.
- Measured: the correct margin against the best other token after the perturbation, and whether the gold token stays the
  masked argmax.
- The primary statistic is the **worst-of-8 flip fraction** at ε = 0.10.
- Gradient sensitivity: `ρ_S` of the preservation gradients.

**Test 4 — breakdown.**
- All results are broken down by category and by stratum: CS English substitutions, translation-like, omissions,
  Mandarin lexical, EN_ONLY, and correct EN/ZH controls.
- Language-direction dominance: on CS A rows, the leverage of SCRIPT1 relative to V, plus `cos(v_j, d)`.

**Statistics.**
- Dialogue-cluster bootstrap: 2,000 resamples, seed 42, over audit dialogues.
- Random-control comparisons use the **maximum over the 5 seeds** of the seed mean, so a weak random baseline cannot be
  cherry-picked. The mean over seeds is also reported.

## 7. P1 verdict (frozen, in order; A rows of the audit partition unless stated)

1. **`P1_INCONCLUSIVE_ENGINEERING_OR_DATA`** if any of these hold:
   - an integrity check fails (smoke, weights, basis checks, or reproduction of `m₀` in the evaluation job within
     1e-3 + 1e-3·|m₀|);
   - fewer than 30 audit A rows;
   - fewer than 4 audit dialogues with A rows;
   - fewer than 15 audit A rows from CS utterances.
2. **`P1_NO_USEFUL_DIRECTIONAL_SIGNAL`** if either of these fails:
   - **R (retention):** mean ρ_V ≥ 10·4/1280 = 0.03125, and the bootstrap 95% lower bound of
     mean ρ_V − max_seed mean ρ_R > 0;
   - **L (leverage, ε = 0.10):** mean Δm_V > 0, the bootstrap 95% lower bound of mean Δm_V − max_seed mean Δm_R > 0,
     and frac(Δm_V > 0) ≥ 0.60.
3. **`P1_SUBSPACE_FEASIBLE`** if R and L hold and all of the following hold:
   - **CS:** on CS A rows, mean ρ_V ≥ 0.03125 and > max_seed ρ_R; mean Δm_V > max_seed Δm_R; frac(Δm_V > 0) ≥ 0.60;
     and, if these rows come from ≥ 4 dialogues, the bootstrap lower bound of the Δm_V − max-seed difference is > 0.
   - **not narrow:**
     - CS A rows that are not `translation_like` or `zh_to_latin` number ≥ 10, and their mean Δm_V > max_seed Δm_R;
     - SCRIPT1 mean Δm on CS A rows < 0.80 · V's.
   - **generalization:** audit mean ρ_V ≥ 0.5 · construction mean ρ_V (A rows).
   - **preservation not concerning:** worst-of-8 flip fraction of V at ε = 0.10 ≤ max(0.10, 2 × the mean over the
     random seeds).
4. **`P1_SUBSPACE_WEAK_OR_NARROW`** otherwise.

`FEASIBLE` only authorizes *proposing* P2. It establishes no ASR improvement.

## 8. Expected runtime (H100 MIG 3g.40gb; each job ≤ 3 h)

| Job | Estimate |
|---|---|
| P0 decode | ≈ 10 min (ACL-0 beam-5: ~1 s per utterance) |
| P1 smoke | ≈ 10 min |
| P1 capture (500 beam-5 decodes plus teacher-forced gradients) | ≈ 30 min |
| P1 leverage and preservation evaluation | ≈ 30 min |
| CPU steps | minutes |

If a smoke projects more than 2.5 h for a job, the run stops and is reported. It is not split silently.

## 9. Stop gates

- P0 not passed → stop, with no P1.
- Smoke failure → stop.
- Data audit invalid → `P0_BLOCKED` / P1 `INCONCLUSIVE`.
- No P2 in this sprint.

## 10. Claim boundaries

P1 never claims any of the following:
- word recovery by a controller;
- free-decoding sufficiency;
- learned localization;
- causal isolation of language interference;
- acoustic–language disentanglement;
- superiority over PEFT;
- lower total training compute.

Separability, retention and oracle controllability are reported as such.

## 11. Artifacts (local, `/mnt/data/tungnx/cs-asr-reft/`)

- `p0/`: `data_audit.json`, `panel.json`, `auto_outputs.jsonl`, `manifest.json`, `output_seal.json`,
  `evaluation.json`, `error_inventory.json`
- `p1/`: `data_audit.json`, `panel.json`, `targets.json`, `smoke.json`, `decode.jsonl`, `gradient_manifest.json`,
  `gradients/rows.jsonl`, `gradients/tensors.pt`, `svd_spectrum.json`, `basis_rank4.pt`, `basis_manifest.json`,
  `controls.pt`, `random_controls.json`, `local_leverage.json`, `preservation_audit.json`, `evaluation.json`,
  `manifest.json`

Only code, configs, tests and sanitized summaries are committed.

## 12. Amendments

(none)
