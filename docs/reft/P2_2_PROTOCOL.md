# P2-2 — on-policy continuation preservation (protocol)

**Status:** pre-registered before any P2-2 coverage statistic, training run or outcome existed. It is frozen in
`p2_2/freeze.json`.

**Files:**
- Code: `reft/p2_2_alignment.py`, `reft/p2_2_train.py`, `reft/p2_2_evaluate.py`.
- Configuration: `configs/reft/p2_2_frozen.yaml`.
- Tests: `tests/reft/test_p2_2.py`.
- Slurm: `slurm/reft_p2_2*.sbatch` and `slurm/reft_p2_2_submit.sh`.
- Rules: `P2_2_DECISION_RULES.md`.

**Type.** A bounded method-development pilot. The exposed dev150 panel selects checkpoints and evaluates them; this is
not confirmation.

**Out of scope:** P2-3, P3, M1, encoder adaptation, LoRA retuning, and any λ, seed, architecture or data change.

## 1. Question and arms

Can training R2 LoReFT on its own generated decoder histories reduce Mandarin switch-back omissions beyond gold-prefix
KL (GoldKL), while keeping its English corrections?

| Arm | Objective | Source |
|---|---|---|
| C0 R2-CE | P2-1 CE | P2-1b selected checkpoints, reused after hash verification |
| C1 R2-GoldKL | CE + 1.0 · gold-prefix KL (teacher-correct mask) | P2-1b, reused |
| **M2 R2-OnPolicy** | CE + 1.0 · GoldKL + **1.0 · L_OP** | new; seeds 42, 43, 44 |

**M2 keeps everything else identical:**
- architecture and parameters: R2 LoReFT at layer 31, rank 2, 5,122 parameters;
- initialization: the same orthogonal init, with `torch.manual_seed(s)` → backbone → adapter. The `base` buffer is
  checked against the matched KL run, and training aborts if it differs;
- data and order: the 500 training utterances, in the order `default_rng(s + epoch)`;
- optimization: AdamW at 5e-4, betas (0.9, 0.99), ε 1e-6, weight decay 0, clip 1.0, micro-batch 8 × accumulation 2,
  including the partial final step, for 3 epochs;
- decoding: frozen Auto-LID and the canonical CUDA 12.8 environment.

**Gold terms.** The gold CE and GoldKL terms are P2-1b `kl_step`, unchanged.

## 2. Rollouts

**When.** At the start of each epoch e, the adapter θ_e is frozen. For epoch 1 that is the identity, so the rollout
equals B0. This is expected and is reported.

**What.**
- θ_e decodes all 500 training utterances with canonical frozen-LID beam-5 (`decode_frozen_lid`).
- Events are built from those actual token sequences.
- Both the rollouts and the events are written once to `training/M2_seed{s}/rollouts/`.

**What is never done:**
- no gradient flows through generation or selection;
- there is no refresh within an epoch;
- no development data is used.

## 3. On-policy events (`reft/p2_2_alignment.py`; training references `p1/targets.json` only)

**Reference locations r**, meaning the next unit to be produced:
- **SB:** the first non-filler Mandarin unit after a substantive English unit.
- **PRE:** the start of that English run.
- **ZH:** the third unit of a non-filler Mandarin run of at least 6 units (a preservation control).

**Event choice.** For each utterance and epoch there is at most one event per category. Candidates are ordered by
`sha256("p2_2-event:seed:epoch:utt:cat:r")`, and the first one whose location validates in the generated hypothesis is
taken. Nothing is substituted if it later lacks a target.

**Location** (with `text.one` canonical alignment of the hypothesis to the reference):
- the ≤ 2 reference units before r are exact matches;
- that context occurs exactly once in the reference;
- a clean token boundary (P2-1c `boundaries` / `anchor_at`) holds exactly the hypothesis units through r − 1.

The prefix is then `[SOT, LID, TRANSCRIBE, NOTIMESTAMPS] + generated tokens[:k]`. It is an actual generated prefix:
nothing is spliced and no hidden state is transplanted. A prefix whose next token is EOS (an omitting path) is
included.

**Target hierarchy:**

| Case | Condition | Target |
|---|---|---|
| **T** | Frozen B0, replayed from exactly that prefix (P2-1c replay semantics: `GenerationMixin.generate`, canonical suppression, begin-suppression pinned at absolute position 4, beam 5, ≤ 32 new tokens), reproduces the next ≤ 8 reference units as exact matches immediately after the prefix. This excludes premature EOS, omission, substitution and derailment. | B0's continuation tokens covering the next ≤ 4 units (clean boundary), capped at 8 tokens. Loss: token-mean **KL(p0 ‖ pθ)**, full vocabulary, float32, teacher detached, same audio and prefix. |
| **R** | T is invalid, and the reference scope (the next ≤ 4 units, verbatim from the training target text) can follow the generated prefix: its standalone tokenization reproduces its bytes, and prefix units + scope units equal the canonical units of the joined text. The separator is one space only between Latin/digit characters; Mandarin follows directly, as in the training targets. | Those tokens, capped at 8. Loss: token-mean **CE**. |
| **U** | Neither T nor R. | No loss. The reason is recorded, and the event stays in every denominator. |

**Label check.** Every T/R event is independently re-aligned: prefix + target must place reference units r, r + 1, …
right after the prefix as matches. Any mismatch is counted and gated.

**Prefix class.** Each event's prefix is compared with B0's prefix at the same location (B0 training rollout):
- `identical_tokens`;
- `format_only` (same canonical units, different tokens);
- `lexical`;
- `b0_unlocated`.

Any class other than `identical_tokens` is **non-identity**.

## 4. Loss

- **Per micro-batch:** `L = kl_step(gold CE + 1.0 · GoldKL) + 1.0 · L_OP`, divided by the accumulation steps.
- **L_OP** is the mean over the micro-batch's active events, each event a token mean, so longer events carry no extra
  weight. It is 0 when no event is active.
- **Logged per epoch:**
  - gold CE, GoldKL, OP-T KL, OP-R CE and the OP batch mean;
  - event and token counts;
  - non-identity events;
  - total gradient norm, and the OP gradient norm (measured as the gradient increment of the OP backward);
  - rollout, event-build, teacher and training time;
  - peak VRAM.

## 5. P2-2A (training-only feasibility and engineering preflight, one GPU job, no training)

**Rollouts.** B0, plus the existing CE42 (P2-1 epoch 2) and KL42 (P2-1b selected) checkpoints, on the 500 training
utterances. Events are built with the epoch-0 order.

**Coverage gate**, evaluated separately on CE42 and on KL42; both must pass:

| Requirement | Minimum |
|---|---|
| Valid (T + R) events | ≥ 30 |
| Utterances | ≥ 20 |
| Dialogues | ≥ 6 |
| Valid SB events | ≥ 20 |
| Valid non-identity-prefix events | ≥ 10 |
| Location-label mismatches | 0 |

Failure → `P2_2_BLOCKED_ONPOLICY_COVERAGE`, and the run stops. The thresholds are never relaxed.

**Reported:** opportunities, located, T, R and U by category and reason, teacher-invalid reasons, B0 dropping under the
own prefix, prefix classes, utterances and dialogues, event lengths, prefix-length quartiles, and the SB share of all
opportunities.

**Engineering gates:**

| Gate | Requirement |
|---|---|
| **G1** | Branch; P2-0, P2-1b and P2-1c chains; control adapter hashes; pinned baseline; weights; no training outputs yet |
| **G2/G3** | The coverage gate and label consistency (prefix construction and target validity are also unit-tested) |
| **G4** | Identity adapter: OP-T KL exactly 0 and OP-R CE finite. Perturbed adapter: OP-T KL > 0, with gradients reaching R, W and b, finite. Backbone gradients none; batched vs single-event loss within 5% (BF16); hooks cleared. **The M2 loop with λ_OP = 0 reproduces P2-1b KL_seed42 epoch 1 bitwise**, and the initial `base` matches. |
| **G5** | Identity training rollouts equal B0 (20/20); identity dev beam-5 and greedy equal the pinned baseline (20/20 each); empty-prefix replay parity with recorded rollouts (10/10); hooks cleared |
| **G6** | Projected run time (3 × [rollout + events + M2 epoch] + dev decodes) ≤ 9000 s per job; total slice-hours reported against the 6-hour target |
| **G7** | Authorization record, frozen rules |

All pass → `READY_FOR_P2_2_TRAINING`. Otherwise → `P2_2_BLOCKED_<gates>`, and nothing is trained.

## 6. P2-2B

**Runs.** `M2_seed42`, `M2_seed43` and `M2_seed44` only.

**Every run:**
- epoch-0 identity (150/150);
- 3 rollout refreshes;
- an adapter saved after each epoch;
- frozen-LID beam-5 and greedy dev decodes each epoch;
- a weights check;
- a manifest.

**Chain** (`slurm/reft_p2_2_submit.sh`, `afterok`, kill on invalid dependency): train → select → evaluate, verify
(reload 20/20 per epoch), mech-anchors → mech-replay → report.

**Selection.** The unchanged P2-1 rule: lowest CS MER (beam-5), then MER, then Mandarin CER, then the earliest epoch.

## 7. Evaluation (CPU; canonical scorer and P2-1b metric code)

**Systems:** B0, C0, C1 and M2, every epoch and both regimes.

**Integrity.** C0 and C1 must reproduce the P2-1b evaluation exactly.

**Metrics:**
- MER, CS MER, English WER, PIER, Mandarin CER;
- English corrections, corruptions and net;
- English fates;
- Mandarin new errors, repairs and retention;
- outside-target damage;
- loops and non-terminations;
- improved and worsened dialogues;
- the P2-1D/P2-1b failure metrics (new deletions, long runs, SB units, near-switch vs far, no-English-repair
  utterances, worst dialogue);
- switch-back span fates (intact ≥ 0.8, partial, dropped ≤ 0.2).

**Comparisons.**
- **Primary:** paired M2 − GoldKL per seed and as the seed mean.
- **Secondary:** M2 − CE.
- Both use 2,000 dialogue-bootstrap draws (seed 42), shared across arms.
- Epoch-matched comparisons are secondary.

**Mechanism audit** (exploratory, outside the verdict):
- the ten P2-1c cases and anchors;
- the P2-1c engine (firewalled), for B0, KL42–44 and M2 at the selected checkpoints;
- conditions: original; continuation from B0's `pre_en`, `post_en` and `early_zh` prefixes; own prefix (the P2-1c T3
  rule applied to M2's outputs); B0 at the own prefix; local next-token Δ; T5 sequence preference;
- the KL rows must reproduce P2-1c.

## 8. Claim boundaries

Not established:
- generalization (one exposed panel of 20 dialogues);
- that on-policy specificity, rather than extra supervision, causes any gain (no matched-supervision control);
- superiority over an optimized LoRA;
- acoustic grounding;
- efficiency superiority. Parameters are unchanged, but training compute is not.
