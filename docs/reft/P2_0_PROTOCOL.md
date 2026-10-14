# P2-0 — Auto-compatible training protocol and engineering preflight (frozen)

**Status.** FROZEN before the P2-0 GPU smoke and before any P2-1 optimizer run.
- Freeze record: `/mnt/data/tungnx/cs-asr-reft/freeze/p2_0.json`.
- Constants: `configs/reft/p2_0.yaml` (smoke) and `configs/reft/p2_1_frozen.yaml` (the P2-1 design). P2-1 starts only after
  the project owner creates `/mnt/data/tungnx/cs-asr-reft/p2_1/AUTHORIZATION.json`.
- Code:
  - `reft/adapters.py`
  - `reft/auto_policy.py`
  - `reft/train_supervised.py`
  - `reft/evaluate_supervised.py`
- Tests: `tests/reft/test_p2_0.py`.

**Research direction.** Acoustic-Conditioned Representation Fine-Tuning for efficient code-switching ASR.
- The frozen-SVD tiny-controller line is discontinued; its verdicts stand:
  - `P1_SUBSPACE_WEAK_OR_NARROW`
  - `P1D_NO_CONVINCING_SIGNAL`
  - `P1E_WHEN_ONLY`
- No P1 V or U basis and no P1 oracle gradient is used.

## 1. Model and site

- **Model.** Whisper-large-v3 at `/mnt/data/tungnx/whisper-large-v3`, BF16, eager attention on all 96 modules, frozen.
  The `model.safetensors` and parameter hashes must equal the P0 anchors.
- **Site.** Decoder layer 31, the final layer, post-cross-attention residual, pre-FFN: `h = q + u`.
  - q is the residual entering the cross-attention block; u = `encoder_attn(encoder_attn_layer_norm(q), H_enc)`.
  - h is the input of `final_layer_norm` and of the FFN residual.
  - The block output is h + FFN(h), and it feeds `decoder.layer_norm`.
- **Position scope.** Because layer 31 is final, an update at decoder position p changes only the logits at p.

## 2. Frozen Auto-LID decoding contract (primary for every method)

**Stage A.** The language is the argmax over Whisper's language tokens of the logits at `<|sot|>` on
`input_features[:, :, :3000]`, computed by the frozen backbone with every adapter **disabled**. This is exactly
`WhisperGenerationMixin.detect_language`.
- No reference transcript or oracle language label is used.
- Non-zh/en detections (ko, ru, fi, …) are kept.

**Stage B.** `generate(language=<detected>, task="transcribe", num_beams=5, do_sample=False, temperature=0.0,
max_new_tokens=200, condition_on_prev_tokens=False)`, batch 1, with the adapter active. Greedy (`num_beams=1`) is
secondary. The prompt is `[<|sot|>, <|lang|>, <|transcribe|>, <|notimestamps|>]`, and the adapter cannot change the
language token.

**Parity proof required (G2).** On all of dev150:
- Stage A equals the native language and the P0 manifest;
- native `generate(language=None)` equals the two-stage frozen decode token for token (beam-5);
- two-stage beam-5 equals the sealed P0 outputs in language, tokens, termination and text hash;
- native greedy equals two-stage greedy on 20 rows;
- the cached training LIDs are recomputed and equal for all 500 training utterances.

The two-stage frozen decode (beam-5 and greedy) is stored as the P2 frozen baseline
(`p2_0/frozen_baseline_dev150_{beam5,greedy}.jsonl`). It must reproduce P0 MER 0.118 and genuine-CS MER 0.121 exactly.

## 3. Teacher-forced objective (identical for every method)

- Prompt: the frozen-LID prompt above, taken from the frozen Auto decode of each utterance (reference-free; the training
  LIDs are recomputed in the smoke).
- Target: `tokenizer(target_text) + [EOS]`, using the ACL-0 `target_text` convention (markers and punctuation removed,
  case kept, no spaces next to Han). The target must end in exactly one EOS, every content id must be < EOS, and the
  maximum length is 191 including EOS. A violation raises an error.
- Decoder input `X = prompt + target[:-1]`, right-padded with EOS.
- Labels: `Y[3 + t] = target[t]`. Positions 0–2 and padding are −100. The `<|notimestamps|>` state (position 3)
  predicts content token 0, and EOS is supervised once.
- Loss: token-mean cross-entropy over the labelled positions on the full vocabulary, never `task="translate"`.
  - 1 of 18,116 training content tokens is in the generation suppress list, so CE trains a token decoding can never
    emit. This is recorded and accepted, with no change to the targets.

## 4. Adapters (P2-1; trainable parameters in float32, backbone frozen)

**R2 — standard LoReFT at the site:** `h̃ = h + Rᵀ(Wh + b − Rh)`, with rank 2.
- R ∈ ℝ^{2×1280} has orthonormal rows through torch's orthogonal parametrization (Householder).
- Initialization: R = QR of a seeded Gaussian (seed 42); W = R and b = 0. The update is computed as
  `((W − R)h + b)ᵀR`, so at initialization it is exactly zero.
- Applied at absolute decoder positions ≥ 3, read from the layer's `cache_position`. This covers the state that predicts
  content token 0 and every content state; prefix states 0–2 are untouched.
- Implemented as u → u + Δ inside layer 31's `encoder_attn` output, so h̃ feeds both the FFN branch and the residual
  path.
- **5,122 parameters.**

**L1 — single-layer cross-attention LoRA:** rank 1 on `decoder.layers[31].encoder_attn.q_proj` and `.v_proj`
(1280×1280 each).
- Update `y = W₀x + b₀ + (α/r) B A x`, with α = 1 (scale 1) and no dropout.
- Initialization: A is Kaiming-uniform (bound 1/√fan_in, seed 42/43) and B = 0. No adapted bias.
- **5,120 parameters.**
- This is a weak single-layer LoRA control, not a strong multi-layer LoRA. It acts inside cross-attention, where q_proj
  reads LN(q) and v_proj reads the encoder states, at every position. LoReFT acts on the post-cross-attention residual.
  The two sites are related but not identical.

**M1 (future, not implemented).** `h̃ = h + Rᵀ(W_h h + W_u u + b − Rh)`, rank 2, 7,682 parameters. Its matched
comparison is R3 (7,683). The LoReFT hook already exposes q and u (`record=True`).

## 5. Data roles

**Training.** 500 utterances: the P1 panel from `loc-train ∪ util-train`, 27 dialogues. That is 391 former construction
and **109 former P1 audit** utterances.
- The audit utterances are now training data and can no longer be P2 evaluation data. The historical P1 audit is
  unaffected.
- Only audio and reference transcripts are used. No P1 gradient or basis.
- Manifest `p2_0/train_manifest.json`, sha256 `db4a3d26…`.

**Development.** P0 dev150 (`p2_0/dev_manifest.json`, sha256 `97f4d24e…`), 20 D-dev-select dialogues.
- It is heavily exposed (A3-ACL0-DEV150, P0, ACL-0) and is used for checkpoint selection under a fixed rule.
- No untouched-test claim is possible on it.

**Disjointness (G1).** 0 shared dialogue, conversation, speaker, corpus speaker, audio hash or utterance ID between
training and dev. Segment-time overlap cannot be checked (no timestamps in the manifest).

**Never opened:** D-test, D-dev-confirm, router-calib and external confirmation corpora. Their dialogues are known from
`assignment.json` only.

## 6. P2-1 design (frozen in `configs/reft/p2_1_frozen.yaml`; not authorized until the owner's AUTHORIZATION.json exists)

**Shared settings:**
- seed 42, 3 epochs;
- micro-batch 8 × accumulation 2 = 16 (32 updates per epoch, 96 in total);
- AdamW, betas (0.9, 0.99), ε 1e-6, weight decay 0, constant learning rate, gradient clip 1.0;
- epoch order `default_rng(42 + epoch)`;
- BF16 backbone, FP32 adapter.

**Learning rates (one per method, no search):**
- R2: 5e-4. This is the ACL-0 rate for a residual editor at the same site and budget (96 updates). LoReFT reference
  implementations use 1e-4 to 1e-3.
- L1: 1e-4, the conventional LoRA rate.

These are starting values, not validated optima. The risk that a rank-1 LoRA with B = 0 moves little in 96 updates is
acknowledged in advance. It will not be tuned on dev.

**Each training job:**
- re-decodes dev150 with the zero-initialized adapter and must reproduce the frozen baseline (150/150 beam-5);
- after every epoch, saves the adapter and decodes dev150 (beam-5 and greedy, frozen LID, LID re-checked with the
  adapter disabled);
- refuses to run without the owner-created `p2_1/AUTHORIZATION.json`.

**Checkpoint selection.**
- Rule: lowest genuine-CS MER (beam-5), then overall MER, then Mandarin CER, then the earliest epoch.
- Every epoch is reported, and R2 and L1 use the same rule.
- This is development-set selection, not confirmation.

**Evaluation** (`reft/evaluate_supervised.py`, P0 scorer conventions). Beam-5 is primary; greedy and teacher-forced loss
are never evidence of success.

| Area | Reported |
|---|---|
| Accuracy | MER, CS MER, English WER, English PIER, Mandarin CER; English-only and Mandarin-only strata |
| English and Mandarin unit changes | English error kinds; substantive English corrections and corruptions against the frozen baseline; new Mandarin errors and repairs; Mandarin retention; outside-target damage |
| Decoding behaviour and spread | loops and non-termination; improved and worsened utterances; dialogues with corrections or corruptions; the largest single-utterance share of the net |
| Uncertainty | dialogue-cluster bootstrap (2,000 resamples, seed 42) of ΔMER and ΔCS-MER |
| Efficiency | trainable parameters; adapter bytes; updates and tokens; GPU time; peak VRAM; decode latency against the baseline |

## 7. Engineering acceptance (P2-0 smoke; one GPU job, ≤ 40 minutes)

| Test | Checks |
|---|---|
| A — model and site | 32 layers, d = 1280, layer 31 is last; h = q + u bitwise; block = h + FFN(h); `decoder.layer_norm` consumes the block output; weights equal P0 |
| G2 — Auto parity | as in §2 |
| B — zero update (both methods) | parameter counts 5,122 / 5,120; no backbone parameter trainable; teacher-forced logits bitwise equal (4-row batch); LID unchanged with the adapter installed (20); beam-5 tokens equal the frozen baseline on 150/150; greedy on 20/20; no hooks left; LoRA targets restored |
| C — alignment and causality | with a random nonzero adapter, cached greedy equals teacher-forced argmax under the suppress masks (≤ 2% mismatches; 1-shifted alignment mismatches ≥ 50%); R2 leaves prefix logits 0–2 unchanged; an update from position p leaves logits before p bitwise unchanged; hook positions observed during greedy and beam-5 decoding are [0, 1, 2, 3] at prefill and +1 per step |
| D — beam and KV cache | beam-5 transition scores vs teacher-forced rescoring with the same adapter (zero and random): max ≤ 0.15 and mean ≤ 0.03 nats; decoding X, Y, X leaks no state; batch vs batch-1 teacher-forced argmax agreement ≥ 0.99; no hooks left |
| E — one optimizer step (4 training rows) | finite gradients at step 0, nonzero on every tensor except LoRA A (which is zero by construction while B = 0); after one step, a second backward gives nonzero gradients on every tensor; the effective update is nonzero; no backbone gradient or trainable flag; frozen logits unchanged; save/load is bitwise; R stays orthonormal |
| F — compute | consistent CUDA libraries (pip cuBLAS/cuBLASLt 12.8); weights unchanged; peak VRAM < 38 GB; projected P2-1 job (3 epochs plus per-epoch dev decodes) < 2.5 h |

## 8. Gates and decision

| Gate | Passes when |
|---|---|
| G1 | the CPU audit is valid |
| G2 | the Auto parity checks pass, and the scored frozen baseline reproduces P0 MER and CS MER exactly |
| G3 | the CPU tests on alignment, masking and EOS pass, the target audit is valid, and E passes |
| G4 | B and E pass |
| G5 | B, C and D pass |
| G6 | F passes |
| G7 | freeze, tests, hashes, save/load and a safe commit and push are all in place |

`READY_FOR_P2_1_TRAINING` requires every gate. Otherwise the decision is `P2_0_BLOCKED_<reason>`, with the minimal
repair. Thresholds are not lowered after a failure. Readiness is engineering only; it says nothing scientific.

## 9. Amendments

(none)

### r1 — 2026-10-14T11:26:16Z — canonical P2 CUDA environment and re-anchored frozen baseline (project-owner approved)

The "(none)" above records the state at the original freeze (`freeze/p2_0.json`, sha256 `1574947e…`, commit
`f4bc3b7`). That file is preserved unchanged. This amendment is chained to it: `freeze/p2_0_r1.json` records its
predecessor's name and hash, and every P2 command verifies the whole chain.

**Evidence.**
- The P2-0 smoke (Slurm 59208) failed only the original G2 requirement "two-stage = sealed P0 tokens": 133/150.
- The controlled diagnostic (Slurm 59212, `p2_0/diag_g2_env_{p0env,p2env}.json`) decoded the 17 differing and 5 control
  utterances under both library resolutions:
  - native Auto = two-stage frozen-LID in both environments (22/22);
  - with the conda env's `libcublasLt.so.12.1.0.26` (the P0 launcher) all 17 reproduce sealed P0;
  - with torch's pip `libcublasLt` 12.8 all 17 equal the P2 decode.
- The drift is therefore a BF16 cuBLASLt kernel difference at beam near-ties, not a decoding-policy difference.
- Metric effect: MER 0.1178 → 0.1199; CS MER 0.1206 → 0.1227.

**Decision (owner-approved).** The corrected CUDA 12.8 environment is the canonical P2 environment: pip
`nvidia/*/lib` first, as in `slurm/reft_p2.sbatch`.
- **Pinned baseline:** the P2 frozen baseline is the two-stage frozen-LID decode produced in that environment by
  smoke 59208, pinned in `configs/reft/p2_1_frozen.yaml` (`frozen_baseline`):
  - beam-5 `734f02e2…`;
  - greedy `e689a2f5…`;
  - scored beam-5 MER 0.1199, CS MER 0.1227.
- **P0 artifacts:** all P0 artifacts, the P0 seal and the P0 verdict remain unchanged as the historical record. The
  17/150 P0 mismatch stays documented evidence.

**Amended G2 contract** (replaces the "two-stage = sealed P0" requirement; nothing else in G2 is relaxed). All of the
following must hold in the canonical environment:
1. Stage-A LID equals the native prompt and the manifest (150/150).
2. Native `generate(language=None)` equals the two-stage frozen-LID decode: beam-5 150/150, greedy 20/20.
3. The training LIDs are recomputed and equal (500/500).
4. The pinned baseline files match their hashes, and a fresh run reproduces them token for token (beam-5 150/150,
   greedy 150/150) without overwriting them.
5. The CUDA libraries are consistent.
6. The scored pinned baseline is on record.

The equality counts against sealed P0 are still computed and reported, but they do not gate.

**Code changes** (all listed in `changed_sources` of `freeze/p2_0_r1.json`):
- the versioned, chained freeze and verification;
- the smoke re-run mode, which writes `p2_0/rerun_<job>_*` and never touches the pinned baseline;
- the amended G2 gate;
- `train` refuses to run outside the canonical environment or if the pinned baseline hashes change;
- `evaluate` checks the baseline hashes;
- `acceptance` composes G1–G7 from the smoke produced under the current freeze;
- `slurm/reft_p2_1_submit.sh`, the afterok submission chain;
- a freeze-chain unit test.

**Unchanged:** methods, adapters, data, objective, optimizer, selection rule, metrics and all other thresholds.

**Re-verification.** A fresh P2-0 smoke and the acceptance step run under this amendment. `READY_FOR_P2_1_TRAINING` is
set only if every gate passes there. P2-1 training still requires the separate `p2_1/AUTHORIZATION.json`.
