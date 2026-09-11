# BASIS-A4 — Full Direction × Depth × Model Atlas (SCIENTIFIC SPEC)

Status: **FROZEN PROTOCOL — pre-run additive frozen-steering atlas.** This document
does not reopen DG-00–DG-08, does not change any locked D-test artifact, and does not
introduce a learned controller. It extends the accepted **BASIS-A3** state
(commit `259f3b4`) to two models and (attempted) three directions, and records one
decisive reconciliation finding: **Conditioning-Avg is REDUNDANT** with the existing
Conditioning direction (§2).

Base state: BASIS-A3 accepted at `259f3b4`
(`BASIS_A3_RAW_COND_SCOPE_DEPTH_SPEC.md`, `BASIS_A3_PROTOCOL_RECONCILIATION.md`).

---

## 0. Scope in one paragraph

Build a frozen additive steering atlas across **two models** (Whisper-large-v3,
Qwen3-ASR-1.7B), the accepted **direction families**, **two scopes** (Global,
Oracle-local / CS-only), and the **three frozen evaluation panels** (CS-Dialogue
D-dev-select 300; SEAME dev-man 50; SEAME dev-sge 50). Full-depth screening dose
`rho = 0.5`. No residualized Local vector is used anywhere. No GPU jobs are launched by
this stage; this is the reconciliation + freeze.

---

## 1. Current Conditioning definition (TRACED FROM CODE, not docs)

The BASIS-A2 layer atlas (`results/basis_frozen_layer_atlas/`, spec
`BASIS_FROZEN_LAYER_ATLAS_SPEC.md`) and BASIS-A3 both use the **same** decoder
Conditioning vector. It is built by the canonical construction path shared with
`experiments/dg03_build_basis.py::_conditioning_and_scale` (assembled by
`src/csasr/directions/steering_basis.py::language_conditioning`); the atlas runner
`experiments/basis_frozen_layer_atlas.py:175-180` stacks the same per-utterance
contrasts and feeds `language_conditioning`. `directions.json` labels it
`"canonical per-utterance prefix contrast"`.

**Exact construction (Whisper-large-v3, decoder site `decoder_post_cross_attn_residual`):**

For each construction utterance `u` on **CS-Dialogue D-construct** (baseline-correct
spans; `n_cond = 125` utterances in the atlas manifest) and each decoder layer `l`:

- Forced language prefix `P(c)` from `build_prefix(processor, language=c, task="transcribe")`
  = `[<|startoftranscript|>, <|c|>, <|transcribe|>, <|notimestamps|>]`, a 4-token prompt.
  `c_E = "en"`, `c_M = "zh"`. `len(P(c_E)) == len(P(c_M)) == plen == 4`.
- Transcript tokens `T(u) = tokenizer.encode(normalize(reference_transcript))[:220]` —
  the **entire normalized reference transcript** (matrix Mandarin + embedded English
  content tokens together), teacher-forced, **identical** under both language conditions.
- Sequence decoded (teacher-forced): `seq(c) = P(c) + T(u) + [EOT]`.
- **Content positions** = `slice(plen, plen + len(T(u)))` — i.e. **all** transcript-content
  token positions, **excluding** the 4 forced prompt/control tokens and **excluding** EOT.
- Record exact-site states `r_l,t(u, c)` at every content position, under each prefix.
- Per-utterance contrast: `Δ_l(u) = mean_{t ∈ content} r_l,t(u, c_E) − mean_{t ∈ content} r_l,t(u, c_M)`.

Aggregate over utterances and normalize:

```
v_cond(l) = normalize( dialogue_balanced_mean_u ( Δ_l(u) ) )
          = normalize( mean over dialogues of ( mean over that dialogue's utterances of Δ_l(u) ) )
```

**Exact mathematical definition:**

```
v_cond(l) = normalize(
    (1/|D|) Σ_{d∈D} (1/|U_d|) Σ_{u∈U_d}
        [ (1/|C_u|) Σ_{t∈C_u} r_l,t(u,c_E) − (1/|C_u|) Σ_{t∈C_u} r_l,t(u,c_M) ]
)
```
where `D` = dialogues in D-construct, `U_d` = utterances in dialogue `d`,
`C_u` = all transcript-content positions of `u` (excl. forced prefix + EOT).

**Answers to the required trace questions:**

| Question | Answer (from code) |
|---|---|
| Which decoder positions pooled | ALL transcript-content token positions of the reference. |
| Only CS/embedded positions? | **No.** No embedded/CS restriction whatsoever. |
| Matrix-language positions used? | **Yes.** Matrix (Mandarin) content tokens are pooled together with embedded English. |
| Prompt/language conditions | Same audio, same reference tokens; only the forced 4-token prefix language differs (`<|en|>` vs `<|zh|>`). |
| Special/prompt/padding exclusions | Forced prefix (SOT, lang, transcribe, notimestamps) excluded; EOT excluded; single-utterance forward so no padding. |
| Token vs utterance weighting | **Per-utterance mean of tokens, then dialogue-balanced mean of utterances** (each dialogue equal weight). |
| Subtraction order | English-conditioned minus Mandarin-conditioned (`c_E − c_M`), per utterance, before aggregation. |
| Normalization | Unit-normalized **after** aggregation (single final normalize). |
| Layer-specific construction | Yes; one independent `v_cond(l)` per decoder layer from that layer's site states. |

---

## 2. Conditioning-Avg definition + REDUNDANCY GATE

**Intended definition (from the A4 task):** for each decoder layer `l`,
`mu_all(l,c) = mean r_l,t(x,c)` over **all** eligible transcript-content positions
(matrix + embedded), excluding forced language prompt/control tokens, BOS/EOS, padding,
and audio-prefix/placeholder states; then
`v_cond_avg(l) = normalize( mu_all(l, c_E) − mu_all(l, c_M) )`.

**Comparison to §1.** The stated *intent* of Conditioning-Avg is to differ from
Conditioning "by averaging language-conditioning differences over ALL transcript-content
hidden states." But the traced §1 implementation **already** averages the conditioning
difference over **all** transcript-content positions (matrix + embedded), with **exactly**
the same eligible-position set, the same `c_E − c_M` subtraction order, the same final
normalization, and the same per-layer construction. The distinguishing feature
Conditioning-Avg tried to introduce (all-position averaging) is the status quo.

The **only** residual difference is the cross-utterance aggregation weighting:

- Current Conditioning (§1): **dialogue-balanced mean of per-utterance token-means**
  (each dialogue equal weight; within a dialogue each utterance equal weight).
- Conditioning-Avg as literally written: a **token-weighted grand mean**
  (`mu_all` = one pooled mean over every content token of every utterance).

These two weightings are not bitwise identical when utterance lengths and per-dialogue
utterance counts vary, but they are the **same direction family with a re-weighted
aggregator** — not a new position set, mechanism, subtraction, or normalization. It is a
trivial aggregation variant of the SAME conditioning direction, not a scientifically
distinct steering direction.

### Redundancy Gate verdict: **REDUNDANT**

Per the A4 redundancy gate ("If the existing Conditioning implementation is already
mathematically equivalent to this definition: STOP … Do not create a fake duplicate
direction"), Conditioning-Avg **collapses onto** the existing Conditioning direction and
is **REDUNDANT**. It is **not** materialized as a separate direction in A4.

- The A4 direction set therefore reduces to **{Raw, Conditioning}** (2 families), not 3.
- **Human-overridable note (OPEN DECISION, not blocking):** if a future study wants to
  test aggregation-weighting sensitivity specifically, the token-weighted grand-mean
  variant may be run as an explicitly labeled *weighting ablation of Conditioning*
  (`v_cond_tokweighted`), never as a distinct "Conditioning-Avg direction." It is out of
  A4 core scope.

---

## 3. Raw (accepted BASIS-A3 semantics — unchanged)

`v_raw` is the frozen English-minus-Mandarin baseline-correct contrast, normalized after
the mean difference, constructed on **CS-Dialogue D-construct only**. SEAME dev data
never constructs any vector.

- **Raw decoder** (`decoder_post_cross_attn_residual`): dialogue-balanced mean of
  per-span baseline-correct (English − Mandarin) exact-site states
  (`v2r3_directions.site_d_contrasts` → `steering_basis.raw_language_contrast`),
  normalized after the mean difference.
- **Raw encoder** (`encoder_post_self_attn_residual_pre_ffn`): mean baseline-correct
  English acoustic representations − mean baseline-correct Mandarin acoustic
  representations, normalized after the mean difference (A3 encoder construction path).
- Model-specific: Whisper vectors from Whisper D-construct; Qwen vectors from Qwen
  D-construct. Never mix hidden spaces (§11 geometry).
- **Residualized Local (`v_local`) is NOT used anywhere in A4.**

---

## 4. Whisper-large-v3 matrix + reuse

Architecture: 32 encoder layers (E0–E31), 32 decoder layers (D0–D31), `d_model = 1280`.
Decoder site `decoder_post_cross_attn_residual` (DG-02 FROZEN); encoder site
`encoder_post_self_attn_residual_pre_ffn` (A3). `NormPreserve`, no depth rescale.

| Direction | Layers | Scopes | rho | Datasets | Status |
|---|---|---|---|---|---|
| Raw | Enc E0–E31 + Dec D0–D31 | Global, Oracle-local | 0.5 | CS, MAN, SGE | **REUSE A3 R1 rho=0.5 — do NOT rerun** |
| Conditioning | Dec D0–D31 | Global, Oracle-local | 0.5 | CS, MAN, SGE | Reuse D24/D26/D27/D31; run only the missing layers |
| Conditioning-Avg | — | — | — | — | **REDUNDANT (§2) — not run** |

**Whisper reuse matrix (rho=0.5):**

- **Raw:** 64 layers × 2 scopes × 3 datasets = **384 cells, all reused** from accepted
  A3 R1 (`results/basis_a3_raw_cond_scope_depth/raw_r1/`). New Raw decode work: **0**.
- **Conditioning:** 32 layers × 2 scopes × 3 datasets = 192 cells.
  - Reusable (subject to exact hash/protocol-compatibility check): **D24, D26, D27, D31**
    → 4 × 2 × 3 = **24 cells** from `…/conditioning/` (rho=0.5, both scopes exist).
  - New decode work: **28 layers × 2 × 3 = 168 cells** (D0–D23, D25, D28, D29, D30).

Reuse acceptance rule: a prior cell is reused only if its resolved protocol hash,
direction hash, panel fingerprint, model revision, dose, `NormPreserve`, and site name
all match the A4 freeze; otherwise it is re-decoded. `Conditioning-Avg` is never decoded.

---

## 5. Qwen3-ASR-1.7B architecture (verified from `config.json` + weight map)

Model `Qwen/Qwen3-ASR-1.7B` at `/mnt/data/tungnx/Qwen3-ASR-1.7B`
(`architectures: ["Qwen3ASRForConditionalGeneration"]`, `model_type: qwen3_asr`,
"thinker" config). It is a **decoder-only causal LM with audio tokens injected in-sequence**,
**not** an encoder–decoder with cross-attention.

**Audio encoder** (`thinker.audio_tower`, `qwen3_asr_audio_encoder`):
- `num_hidden_layers = 24` (E0–E23), `d_model = 1024`, `encoder_ffn_dim = 4096`,
  `encoder_attention_heads = 16`, `num_mel_bins = 128`, `max_source_positions = 1500`,
  `n_window = 50`, `n_window_infer = 800`, `conv_chunksize = 500`.
- Whisper-style layer modules: `self_attn_layer_norm` (pre-attn LN), `self_attn`
  (q/k/v/out_proj), `final_layer_norm` (pre-FFN LN), `fc1`/`fc2` (GELU).

**Projection / modality bridge:** `thinker.audio_tower.proj1`, `thinker.audio_tower.proj2`
(with `downsample_hidden_size = 480`, `output_dim = 2048`). **NOT a Transformer encoder
layer — never counted as one. Encoder depth = 24.**

**Text decoder** (`thinker.model`, `qwen3` causal stack):
- `num_hidden_layers = 28` (D0–D27), `hidden_size = 2048`, `intermediate_size = 6144`
  (SwiGLU), `num_attention_heads = 16`, `num_key_value_heads = 8` (GQA), `head_dim = 128`,
  `hidden_act = silu`, `rms_norm_eps = 1e-6`, RoPE with mrope (`interleaved`,
  `mrope_section = [24,20,20]`), QK-norm (`self_attn.q_norm`, `self_attn.k_norm`).
- Layer modules: `input_layernorm`, `self_attn` (q/k/v/o_proj + q_norm/k_norm),
  `post_attention_layernorm`, `mlp` (gate/up/down_proj). Final `thinker.model.norm`.
- Audio special tokens (in the decoder sequence): `audio_start_token_id = 151669`,
  `audio_end_token_id = 151670`, audio placeholder `audio_token_id = 151676`;
  EOS `[151643, 151645]`, pad `151643`.
- Feature extractor: `WhisperFeatureExtractor`, 128 mel, hop 160, n_fft 400,
  chunk_length 30s (`preprocessor_config.json`).

---

## 6. Exact intervention sites (frozen tensor definitions)

Whisper (unchanged from DG-02 / A3):

- **Whisper decoder** `decoder_post_cross_attn_residual` = `residual + encoder_attn(...)`,
  the pre-FFN tensor (`METHOD_CONTRACT §1`; `csasr.lss.sites`). Reconstructed from a
  pre-hook on `encoder_attn_layer_norm` (residual) + a forward hook on `encoder_attn`.
- **Whisper encoder** `encoder_post_self_attn_residual_pre_ffn` = `q_enc + self_attn(...)`
  immediately before the encoder FFN (A3).

Qwen3-ASR (new — **Whisper cross-attention semantics are NOT reused**):

- **Qwen audio-encoder site** `qwen_audio_encoder_post_self_attn_residual_pre_ffn`.
  In each `thinker.audio_tower.layers[l]`:
  ```
  residual = hidden_states
  h = self_attn_layer_norm(hidden_states)
  h = self_attn(h)
  SITE_ENC_l = residual + h        # <-- post-self-attn residual, pre-FFN (input to final_layer_norm)
  ```
  Reconstruct via a forward pre-hook on `final_layer_norm` (captures `SITE_ENC_l`), or
  equivalently residual (pre-hook on `self_attn_layer_norm`) + `self_attn` output.

- **Qwen text-decoder site** `qwen_text_decoder_post_self_attn_residual_pre_mlp`.
  In each `thinker.model.layers[l]`:
  ```
  residual = hidden_states
  h = input_layernorm(hidden_states)
  h = self_attn(h)                 # causal, RoPE, GQA, QK-norm
  SITE_DEC_l = residual + h        # <-- post-self-attn residual, pre-MLP (input to post_attention_layernorm)
  ```
  Reconstruct via a forward pre-hook on `post_attention_layernorm` (captures `SITE_DEC_l`).

Both Qwen sites use the same `apply_steering(..., norm_preserve=True)` kernel and **no**
depth rescale, mirroring the frozen Whisper repair. Steering is added in the exact space
it is constructed in. `g=0`/`β=0` invariants hold.

---

## 7. Qwen conditioning mechanism (frozen BEFORE any result)

Verified from the official `qwen_asr` package
(`inference/qwen3_asr.py::_build_messages` / `_build_text_prompt`, `inference/utils.py`).
Language is set through the **assistant forced prefix**, not a Whisper-style special token.

Chat layout (from `chat_template.json`):
```
<|im_start|>system
{context}<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
language {LANG}<asr_text>{transcript…}
```
`_build_text_prompt` appends the literal `f"language {force_language}<asr_text>"` after the
generation prompt. `force_language` is canonicalized (`normalize_language_name`) to
`"English"` / `"Chinese"`.

**Qwen conditioning construction (`v_cond_qwen(l)`, text-decoder site), analogous to §1:**

- Same audio and same reference transcript tokens `T(u)`, teacher-forced.
- Change **only** the forced language tag: `c_E → "language English<asr_text>"`,
  `c_M → "language Chinese<asr_text>"`; `context = ""` (empty system prompt) in both.
- **Content positions** = the reference transcript token positions **strictly after
  `<asr_text>`** and **before the first EOS**. Excluded from pooling: system/user template
  tokens, `<|im_start|>/<|im_end|>`, audio-placeholder states (`<|audio_start|>`, expanded
  audio embeddings, `<|audio_end|>`), the `language {LANG}<asr_text>` control tokens, and EOS.
- `Δ_l(u) = mean_content r_l,t(u,c_E) − mean_content r_l,t(u,c_M)`;
  `v_cond_qwen(l) = normalize( dialogue_balanced_mean_u Δ_l(u) )` — identical aggregator to §1.

**Canonical / neutral Qwen baseline inference policy (frozen, not tuned post-hoc):**
- Transformers backend, greedy: `do_sample=False`, `temperature≈0` (per
  `generation_config.json`), beam 1, `context=""`.
- `force_language="Chinese"` (matrix language forced), mirroring the Whisper A3 baseline
  `language=zh`, so baseline and `c_M` share the language condition. `max_new_tokens=200`
  to match Whisper.
- **OPEN DECISION (documented default):** `force_language="Chinese"` is the frozen default;
  `force_language=None` (auto-LID) is registered as a sensitivity-only variant and is not
  used for headline A4 cells. This choice is frozen **before** any scientific decode.

---

## 8. Qwen matrix + Global/Local masks

| Direction | Layers | Scopes | rho | Datasets |
|---|---|---|---|---|
| Raw | Audio-enc E0–E23 + Text-dec D0–D27 | Global, Oracle-local | 0.5 | CS, MAN, SGE |
| Conditioning | Text-dec D0–D27 | Global, Oracle-local | 0.5 | CS, MAN, SGE |
| Conditioning-Avg | — | — | — | REDUNDANT (§2) |

**Masks (both models):**

- **Encoder Global:** all valid non-padding acoustic frames.
- **Encoder Oracle-local:** frames for reference embedded-English acoustic spans. Reuse the
  **same BASIS-A3 reference spans in acoustic time (seconds)**, mapped to **each model's
  actual encoder frame grid** using that model's measured frames-per-second (Whisper ≈50 Hz
  encoder grid as in A3; Qwen audio-tower fps measured empirically from encoder output
  length vs audio duration at construction and **frozen before results** — not assumed equal
  to Whisper). Padding frames always zero-gain. A3's 80 ms boundary tolerance is retained.
- **Decoder Global:** all eligible transcript-generation states. **Never modified:**
  language/prompt tokens, audio-prefix/placeholder states (Qwen: `<|audio_*|>` + expanded
  audio embeddings; the `language …<asr_text>` control tokens), padding, special/EOS tokens.
- **Decoder Oracle-local:** reference-aligned embedded-English transcript-generation
  positions only.

---

## 9. Cross-model fairness logging (rho=0.5, no per-model rho tuning)

`rho=0.5` is the same nominal dose but not the same physical perturbation across
architectures. **Do NOT tune model-specific rho in A4.** For **every** decoded condition,
log: original hidden-state norm, mean perturbation norm, total intervention energy,
relative perturbation (‖u‖/‖r‖), edited-position count, and edited fraction. Matched-energy
follow-up is future work.

---

## 10. Geometry (within-model only)

For every decoder/text layer, **within each model separately**, compute Raw vs
Conditioning (Conditioning-Avg is redundant, so Raw↔Cond-Avg and Cond↔Cond-Avg
comparisons are dropped). Metrics: cosine, angle, raw L2, unit-normalized L2, and vector
norms. **Never** compare vector coordinates across Whisper (1280-d) and Qwen (2048-d)
hidden spaces.

---

## 11. Metrics + data exposure

Reuse canonical A3 `metrics_v1`/`result_v1`: MER, PIER, EN-WER, matrix CER,
`poi_corrections`, `poi_corruptions`, `poi_net_utility = poi_corrections − poi_corruptions`,
matrix retention, embedded retention, outside harm where valid, substitutions/deletions/
insertions, intervention energy, edited count. No new metric implementation.

Data exposure (unchanged): CS `D-construct` = vector construction only; CS `D-dev-select`
(300) = evaluation; CS `D-dev-confirm` and `D-test` = **DO NOT USE**. SEAME `dev-man` = 50,
`dev-sge` = 50 frozen panels, **evaluation only, never construction**. No panel expansion.

---

## 12. Expected configuration counts (rho=0.5)

| Model | Direction | Cells | New decode work |
|---|---|---|---|
| Whisper | Raw (E0–E31 + D0–D31) × 2 scopes × 3 sets | 384 | 0 (all A3-reused) |
| Whisper | Conditioning (D0–D31) × 2 scopes × 3 sets | 192 | 168 (reuse D24/26/27/31 = 24) |
| Whisper | Conditioning-Avg | 0 | 0 (REDUNDANT) |
| Qwen | Raw (E0–E23 + D0–D27) × 2 scopes × 3 sets | 312 | 312 |
| Qwen | Conditioning (D0–D27) × 2 scopes × 3 sets | 168 | 168 |
| Qwen | Conditioning-Avg | 0 | 0 (REDUNDANT) |
| **Totals** | | **1056 atlas cells** | **648 new steered cells** |

Plus: Qwen baselines (3 datasets, unsteered free-decode); Whisper baselines reused from
A3; Qwen direction construction (one CS D-construct pass: Raw enc E0–E23, Raw dec D0–D27,
Conditioning dec D0–D27); within-model geometry (CPU).

Had Conditioning-Avg been treated as distinct it would have added
`(32 + 28) × 2 × 3 = 360` decode cells; the redundancy finding removes all of them.

---

## 13. Scientific risks

1. **Conditioning-Avg redundancy is the headline reconciliation result** — reporting a
   third direction would be a fabricated duplicate. Mitigation: dropped; documented with an
   overridable weighting-ablation escape hatch (§2).
2. **Cross-model dose non-comparability** at fixed `rho=0.5` (different site-norm scales
   `s_l`, hidden dims). Mitigation: mandatory energy logging (§9); no rho tuning; matched
   energy deferred.
3. **Qwen has no cross-attention** — the DG-02 Whisper site does not exist. Mitigation:
   Qwen sites are self-attention-residual, tensor-frozen (§6); never relabeled as the
   Whisper site.
4. **Qwen encoder frame-grid mismatch** vs Whisper. Mitigation: map A3 spans in acoustic
   seconds → each model's measured fps, frozen before results (§8).
5. **Qwen language conditioning is prompt-suffix, not a special token** — wrong exclusion
   of the `language …<asr_text>` control tokens would leak the conditioning signal into the
   pooled content. Mitigation: content = positions strictly after `<asr_text>`, before EOS,
   with all control/audio/template tokens excluded (§7).
6. **Qwen baseline language policy** could be tuned to flatter results. Mitigation: frozen
   `force_language="Chinese"` before any decode; auto-LID is sensitivity-only (§7).
7. **Reuse contamination** — reusing A3 cells whose protocol/hash drifted. Mitigation:
   strict per-cell hash/fingerprint/site/dose match or re-decode (§4).
8. **SEAME panels underpowered** (50 each) and exploratory development evidence only — no
   held-out test claim; SEAME never constructs vectors (§11).

---

## 14. Protocol Status: **PASS**

All construction semantics, sites, masks, conditioning mechanisms, reuse rules, counts,
and the redundancy verdict are determined and frozen. Execution plan in
`BASIS_A4_EXECUTION_PLAN.md`. No GPU work is launched by this stage.
