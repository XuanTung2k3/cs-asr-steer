# P2-1b — paired CE vs KL-preservation feasibility (protocol)

**Status.** Frozen before any P2-1b training. The freeze record is `/mnt/data/tungnx/cs-asr-reft/p2_1b/freeze.json`. It
is independent of the P2-0 freeze chain, which is only verified, never extended.

**Files.**
- Constants: `configs/reft/p2_1b_frozen.yaml`.
- Decision rules: `docs/reft/P2_1B_DECISION_RULES.md`.
- Code: `reft/p2_1b_train.py`, `reft/p2_1b_evaluate.py`, `reft/p2_1b_prefix_audit.py`.
- Tests: `tests/reft/test_p2_1b.py`.
- Slurm: `slurm/reft_p2_1b*.sbatch`, `slurm/reft_p2_1b_submit.sh`.

**Type.** Exploratory preservation-feasibility pilot on the exposed dev150 panel, which also selects checkpoints. It is
not confirmation.

**Out of scope:** P2-2, M1, encoder adaptation, LoRA retuning, λ, seed or method sweeps.

## 1. Question

Does frozen-backbone KL preservation reduce R2's **switch-back Mandarin span omissions** (found in P2-1D) while
retaining its English corrections?

The competing outcomes are:
- H1: rescued;
- H2: corrections suppressed;
- H3: the gold-prefix KL misses the free-decoding failure;
- H4: seed variance.

## 2. Arms (paired by seed)

| Arm | Loss | Everything else |
|---|---|---|
| R2-CE_s | P2-1 token-mean CE (labels at 3 + t; prefix 0–2 and padding ignored) | identical within each seed |
| R2-KL_s | CE + λ·KL, λ = 1.0 (fixed, not tuned) | identical within each seed |

**Within each seed s ∈ {42, 43, 44}, the arms share:**
- LoReFT rank 2 at decoder layer 31, 5,122 parameters;
- the init seed s;
- the epoch order `default_rng(s + epoch)`;
- the 500-utterance manifest and the frozen Auto-LID prompts;
- AdamW (lr 5e-4, betas (0.9, 0.99), ε 1e-6, weight decay 0, clip 1.0);
- 3 epochs, micro-batch 8 × accumulation 2, giving 32 updates per epoch including the final partial step;
- BF16 backbone and FP32 adapter;
- the canonical CUDA 12.8 environment.

**The CE path is the P2-1 path.** The CE arm calls `reft.train_supervised.batch_loss` unchanged.

**Historical R2-CE_42 (P2-1).**
- **Reuse condition:** the bounded equivalence gate passes, meaning that one full epoch of the new CE path at seed 42
  reproduces `p2_1/R2/epoch1/adapter.pt` bitwise.
- **Otherwise:** a fresh CE_42 control is trained and disclosed.
- **Run limit:** at most 6 new training runs.

## 3. KL objective

- **Teacher.** The same Whisper forward (same audio features, frozen-LID prompt and gold-prefix decoder inputs) with the
  adapter absent. It is detached.
- **Loss.** `KL(p0 ‖ pθ) = Σ_v p0 (log p0 − log pθ)` over the full vocabulary, using float32 log-softmax of the same
  logits as CE.
- **Mask.** `M = (Y ≠ −100) ∧ (argmax_v z0 = Y)`. This is training-only teacher-forced correctness over content tokens
  and EOS. It is not free-decoding correctness, and no dev or P2-1D information enters it.
- **Normalization.** `L_KL = Σ_M KL / max(1, |M|)` per micro-batch. An empty mask contributes zero and is counted.
- **Total.** `L = L_CE + 1.0 · L_KL`, divided by the gradient-accumulation steps as in P2-1.
- **Logged separately:** CE, KL, total loss, gradient norm, mask count, and the teacher-forward time.

## 4. Preflight (one bounded GPU smoke and CPU tests)

**G1 — provenance.**
- Branch is `cs-asr-reft`.
- The P2-0 chain (`p2_0.json → p2_0_r1.json`) verifies.
- P2-1 and P2-1D artifacts are unchanged (hashes recorded).
- Pinned baseline and manifest hashes match.
- Locked roles are never opened.
- No conflicting jobs or output directories.

**G2 — CE equivalence.** Init, optimizer and order are identical; one epoch of seed-42 CE is compared against
P2-1 epoch 1 (bitwise reuse condition).

**G3 — KL objective:**
- the teacher equals the frozen forward bitwise;
- no teacher or backbone gradient;
- zero-init student logits equal the teacher bitwise, and the initial KL is ≈ 0;
- the CE value and gradient are identical to the CE path at initialization;
- KL ≥ 0 and becomes positive under a nonzero adapter;
- KL gradients reach R, W and b;
- everything is finite;
- normalization is independent of padding and batch shape;
- no hooks are left;
- the label contract holds.

  Synthetic KL-direction tests run on CPU.

**Mask coverage (training data only).**
- Eligible positions, teacher-correct positions and the selected fraction.
- Coverage by token language through `token_unit_map`: EN, ZH, other, unmappable.
- Coverage of switch-back tokens (the first Mandarin unit after a substantive English unit) and of EOS.
- **Stop rule:** if switch-back token coverage is < 0.10, stop for a design decision. No reweighting.

**G4 — Auto decoding.**
- The adapter code is unchanged from P2-0 r1, where positions, KV cache and beam were verified.
- The smoke repeats zero-init parity on 20 dev utterances and a random-adapter cached-greedy vs teacher-forced argmax
  check.
- Every training job reproduces B0 on 150/150 before training.

**G5 — resources.** CE and KL micro-batch times, teacher-forward overhead, and peak allocated and reserved memory. The
projected job must be < 2.5 h.

**G6 — reproducibility.** Seeds, write-once outputs, checkpoint reload, hashes, tests.

**G7 — protocol.** This file, the decision rules, the frozen prefix-audit cases and the dependency graph are frozen
first.

If all pass: `READY_FOR_P2_1B_EXECUTION`. Otherwise: `P2_1B_BLOCKED_PREFLIGHT_<REASON>`.

## 5. Execution

**Training jobs** (one per run, unique write-once directories `p2_1b/training/{CE,KL}_seed{s}`):
- CE_43, CE_44, KL_42, KL_43, KL_44;
- CE_42 only if equivalence fails.

**Dependency chain** (`afterok`, `--kill-on-invalid-dep=yes`):
- training → selection;
- selection → evaluation → reload verification;
- selection → prefix audit;
- verification and prefix audit → report.

**Every run:** epoch-0 zero-init identity (150/150), per-epoch dev decodes (beam-5 and greedy, frozen LID with LID
re-checked), the adapter saved each epoch, and a manifest with losses and efficiency.

## 6. Evaluation

Everything uses the canonical scorer from P2-1 and P2-1D, against the same B0, for every epoch of every run.

**Quality:**
- MER, CS MER, English WER, PIER, Mandarin CER;
- English corrections and corruptions;
- Mandarin new errors, repairs and retention;
- outside-target damage;
- loops and non-termination;
- improved and worsened utterances and dialogues.

**Failure-specific (P2-1D definitions):**
- new Mandarin deletions;
- dropped runs, and long runs (≥ 5);
- long runs starting right after substantive English, with their units (SB);
- damage rate near switches (distance ≤ 7) vs far;
- damage in utterances with no English repair;
- worst-dialogue share.

**English transitions.** B0 → system category fates: omission, translation-like and lexical substitution → correct;
correct → wrong; wrong → a different wrong category.

**Statistics.**
- Per seed for CE and for KL; matched Δ_s = KL_s − CE_s; the mean and range across seeds.
- Dialogue-cluster bootstrap (2,000 resamples, seed 42) with the **same dialogue draws** for CE and KL, both for each
  seed and pooled as the seed mean. The intervals are descriptive.
- All 20 dialogues and the CS subset.

**Efficiency.** Parameters, adapter bytes, GPU job time, pure training time, teacher overhead, peak
allocated/reserved memory, GPU-hours, updates, tokens, decode latency.

## 7. Prefix diagnostic (`reft/p2_1b_prefix_audit.py`)

**Cases** (fixed before training from P2-1D data only; `p2_1b/prefix_audit/cases_private.json`):
- every damaged switch-back run of R2-CE_42 at epoch 2;
- the same number of safe switch-backs, chosen deterministically.

**Systems:** B0, R2-CE_42 at epoch 2, and the selected CE_s and KL_s checkpoints of every seed.

**Measured:**
- **Gold prefix** (the dev reference target in the training-target format, frozen-LID prompt):
  - log p(gold) of the first Mandarin token after the English span, and its margin against the best other token and
    against EOS;
  - whether the argmax changes;
  - the mean gold log-probability over the switch-back span's tokens.
- **Generated prefix:**
  - the longest common token prefix of the system's and B0's beam-5 outputs;
  - at the divergence point, both models' log-probabilities of the two observed next tokens and of EOS, and the script
    of each;
  - whole-sequence teacher-forced rescoring of both candidate outputs under both models (summed and length-normalized),
    as a check on beam effects.

Classification into A, B or C follows the decision-rules document. It is descriptive only, and it is not acoustic
causality.

## 8. Claim boundaries

The pilot does not establish:
- generalization;
- superiority over an optimized LoRA;
- acoustic grounding;
- deployability;
- confirmation.

Three seeds on the same 20 dialogues do not add independent dialogues.
