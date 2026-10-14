# CS-ASR-ReFT P0/P1 protocol — frozen subspace feasibility

**Status: FROZEN** before any P0 decode, before any P1 decode or gradient capture, and before any reference access.
- Freeze records are written once under `/mnt/data/tungnx/cs-asr-reft/freeze/` (`p0.json`, `p1.json`). Each holds
  source hashes and the git head.
- Constants are in `configs/reft/p0p1.yaml`.
- Amendments are appended in §12, timestamped and justified, and only before the outcome they could affect.

> **Active P1 site: decoder layer 31 (final layer), post-cross-attention residual, pre-FFN — see AMENDMENT L31 (§12,
> 2026-10-14T07:41:13Z).** The layer-24 text in §§4–5 is the original frozen specification, kept as a record. Every other P1 rule is
> unchanged.

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

### r1 — 2026-10-14T06:02:22Z — P0 audio-identity check used the wrong hash convention (engineering only)

- **What happened.** P0 evaluation attempt 1 (Slurm 59119) returned `P0_BLOCKED`. Every integrity check passed except
  `audio_sha_ok`, which was 0/150. The decode-time check compared a whole-file sha256 with the role manifest's
  `audio_sha256`. That field uses the `csasr.utils.hashing.sha256_file(path, max_bytes=65536)` convention:
  sha256(str(size) + first 65,536 bytes), per `configs/data/cs_dialogue.yaml` `audio_hash_bytes`.
- **Verification.** Recomputing one file in the manifest convention reproduces its manifest value. A uniform 0/150 is the
  signature of a convention mismatch, not of changed audio.
- **Change.**
  - `audio_hash_ok` now uses the manifest convention (config `data.audio_hash_max_bytes`) in the P0 decode and the P1
    capture.
  - A CPU step `verify-audio` re-checks all 150 decoded files in that convention. It also checks that each file's mtime
    predates the decode job. The evaluation uses this check for the audio-identity integrity item.
- **What is not changed.** No decode is rerun, and no output, metric, threshold, gate or verdict rule changes.
- **Records.** Attempt 1's evaluation and inventory are kept as
  `p0/evaluation_attempt1_BLOCKED_audio_hash_convention.json` and
  `p0/error_inventory_attempt1_BLOCKED_audio_hash_convention.json`.
- **Timing.** This amendment comes after the attempt-1 outcome was displayed. It is purely an integrity-instrument fix
  and cannot change any P0 quantity.

### r2 — 2026-10-14T06:04:47Z — P1 data audit crashed on a missing key (engineering only, before any P1 outcome)

- **What happened.** `data_audit.py p1` (Slurm 59123) raised `KeyError: 'corpus_speaker_id'` while comparing the new
  panel with the P0 panel. The P0 panel file does not carry that column.
- **Fix.** `cross()` compares only the keys present in both row sets. The P1 panel is still compared with the full
  D-dev-select role on all six identity keys.
- **What is not changed.** Sampling, split, gates and thresholds. No P1 output existed.


### r3 — 2026-10-14T06:08:17Z — float32 cuBLAS vector routines abort on the MIG slice (engineering only, before any P1 outcome)

- **What happened.** The P1 smoke (Slurm 59126) aborted. A gdb rerun (Slurm 59127, diagnostic only, no outputs) shows
  `SIGFPE` inside `cublasSdot_v2`, called from a 1-D CUDA `q @ r` in the finite-difference check.
  - The same jobs log cuBLASLt `CUBLAS_STATUS_NOT_INITIALIZED` for float32 GEMMs, which fall back to the legacy path,
    as ACL-0 also recorded.
  - Full float32 forward and backward passes completed.
- **Fix.** Gradients, site activations and every perturbation direction leave the GPU immediately. All vector algebra
  (dots, projections, cosines, direction construction in float64) runs on CPU. Only full forward and backward passes run
  on the GPU.
- **What is not changed.** No quantity, row rule, budget, basis policy or threshold. The smoke's finite-difference and
  bitwise checks still validate the GPU numerics.

### r4 — 2026-10-14T06:20:36Z — CUDA library shadowing in the P1 GPU environment (engineering only, before any P1 outcome)

- **What happened.** Capture attempt 59129 completed the BF16 decode of all 500 rows. It then failed in the float32
  backward with `CUBLAS_STATUS_NOT_INITIALIZED` (`cublasSgemm`), after a run of float32 cuBLASLt init warnings.
- **Root cause.** The launcher prepends the env `lib/`, which is needed for libstdc++. That directory holds
  `libcublasLt.so.12` and `libcudart.so.12` from CUDA 12.1. They shadow torch's pip CUDA 12.8 libraries:
  `nvidia-cublas-cu12` 12.8.4.1 is the version gdb shows loaded as `libcublas`. The cuBLAS/cuBLASLt version mix
  explains:
  - the float32 Lt init failures;
  - the `cublasSdot` SIGFPE (r3);
  - this Sgemm failure.
- **Fix.**
  - `slurm/reft_p1.sbatch` puts every pip `nvidia/*/lib` directory before the env `lib/`.
  - The smoke, capture and leverage steps record the mapped CUDA library paths.
  - The smoke fails unless `libcublas` and `libcublasLt` (and every mapped CUDA library) resolve to a single path each,
    from the same package tree.
  - The smoke is rerun under the fixed environment, and the P1 decode is redone in the new capture job. The attempt-1
    decode is kept as `p1/decode_attempt1_59129_libmismatch_SUPERSEDED.jsonl`; no reference-based quantity was computed
    from it.
- **P0 note.** P0 (59116) ran with the same library resolution as ACL-0 and reproduced sealed ACL-0 B0 beam-5 token for
  token. P0 is not rerun.
- **What is not changed.** No quantity, rule, budget, basis policy or threshold.

### AMENDMENT L31 — Single Final-Decoder-Layer Intervention (2026-10-14T07:41:13Z)

**Decision (project owner).** The primary P1 representation-intervention site is frozen at Whisper-large-v3 decoder
layer **31**, 0-indexed, the final decoder layer. The site is the post-cross-attention residual immediately before the
FFN:

`h = q + encoder_attn(encoder_attn_layer_norm(q), H_enc)`

This is the tensor that enters `final_layer_norm` and also forms the FFN residual. The intervention is
`h_new = h + δ`. It is not the layer-31 post-FFN block output, not the `decoder.layer_norm` output, and not the logits.
This supersedes the layer-24 choice in §§4–5.

**Why layer 31.** The final layer is closest to lexical token prediction and minimizes the downstream decoder
transformation between intervention and logits. It is a research-design choice, not an empirically established
optimum:
- it has the same parameter count as layer 24 at equal width and rank;
- a single late-layer intervention can fail despite strong local gradient leverage;
- late-layer gradients may reflect token or language preference rather than recoverable acoustic information.

No claim is made that layer 31 is optimal or more efficient.

**Previous specification.** Layer 24, same site semantics, rank 4, same rows, weights, SVD, controls, budgets and gates.

**Execution stage when the amendment was received.**
- P0: complete (`P0_PASS_RELEVANT_OPPORTUNITIES`). It is unchanged and not rerun.
- Layer-24 P1:
  - smoke passed (59134);
  - main capture complete (59135);
  - basis built (59136, sha256 `c76b7566…`);
  - held-out evaluation never completed. Leverage job 59137 crashed while serializing a `-inf` margin, after 13:46 of
    GPU time, before writing any result. Analysis job 59138 never ran and was cancelled.

**What had been observed.**
- Seen: the layer-24 pre-evaluation quantities — row coverage and the construction-partition (in-sample) SVD spectrum.
  The top-4 directions held 18% of the weighted energy, with an effective rank of about 104.
- Not computed or observed: any held-out feasibility outcome (retention, leverage, preservation or verdict).

Layer 31 is therefore **prospective with respect to every P1 feasibility outcome**. It is not a choice made after
comparing layers. The layer-24 evaluation is **not completed**, so no 24-vs-31 comparison exists and none will be used
for selection.

**Artifacts.**
- **Kept and valid (layer-independent, reused after hash checks):**
  - P0 (all);
  - `p1/data_audit.json`, `p1/panel.json`, `p1/targets.json`;
  - the layer-independent F1 Auto beam-5 decode `p1/decode.jsonl` (sha256 `488d3783…`, Slurm 59135, consistent CUDA
    libraries). The teacher-forced rows A/B/P depend only on that decode and the unmodified float32 forward, so the
    layer-31 row population must equal layer 24's.
- **Historical (layer 24, untouched in `p1/`):** `smoke_59128/59134.json`, `gradients/`, `gradient_manifest.json`,
  `manifest_capture.json`, `basis_rank4.pt`, `basis_manifest.json`, `controls.pt`, `svd_spectrum.json`,
  `random_controls.json`. These are never relabeled and never mixed into a layer-31 SVD.
- **New (layer 31):** everything under `p1/l31/`. Freeze stage `p1_l31`.

**Repeated checks for the site.** Layer-24 validation does not count for layer 31. The layer-31 smoke reruns every
acceptance item at layer 31:
- exact site identity;
- placement immediately before the final FFN (block output = site + FFN(site); the block output, not the site, feeds
  `decoder.layer_norm`);
- zero-δ bitwise identity;
- position locality (logits before t unchanged);
- greedy cached-vs-teacher-forced indexing;
- shape and finiteness;
- autograd against finite difference;
- ± gradient perturbation;
- weights unchanged;
- bitwise-repeatable gradients;
- consistent CUDA libraries.

A tiny-model CPU test covers the last-layer placement as well.

**Engineering fix included.** In the leverage step, `gold_vs_best_other` is now serialized as null, with
`correct_suppressed` flagged, when the correct token is in Whisper's suppress set (masked `-inf`). This is the 59137
crash. No metric definition changes.

**New jobs.** One layer-31 smoke, one capture that reuses the decode, so its GPU part is teacher-forced only, the basis
(CPU), one leverage evaluation and the analysis (CPU). Estimated GPU time is about 20 minutes; the leverage step took
about 14 minutes at layer 24. The extra GPU compute already spent on layer 24 was about 25 min (smoke 59134, capture
59135, leverage 59137).

**Unchanged.**
- rank 4;
- row eligibility;
- competitor rules;
- weighting and normalization;
- basis policy;
- controls (seeds 42–46, PCA4, SCRIPT1, FULL ceiling);
- budgets ε ∈ {0.05, 0.10, 0.25}, primary 0.10;
- all P1 gate thresholds;
- seed and population.

There is no layer or rank search. The gate is not changed because layer-31 gradients are stronger or weaker.

**Final-layer diagnostic.** Already in the frozen analysis, it is reported explicitly for:
- CS lexical errors;
- EN_ONLY;
- ZH_ONLY;
- translation-like;
- English→English substitutions;
- correct EN and ZH controls.

The SCRIPT1 language-direction dominance test decides whether leverage is a generic language-prior shift.

**P2 consistency (recommendation only, not implemented).** If P1 passes, P2 uses exactly this site:
- layer 31, post-cross-attention, pre-FFN;
- frozen V₃₁, rank 4;
- one projected-state controller.

LoReFT is compared at the equivalent site. LoRA comparisons must document their target projections. Matched parameter
counts alone do not establish a fair comparison.

