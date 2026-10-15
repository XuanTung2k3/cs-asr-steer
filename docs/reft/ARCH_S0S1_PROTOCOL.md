# S0/S1 — parameter-matched R3 vs M1 under GoldKL (protocol)

**Status:** pre-registered before any R3 or M1 training outcome existed. It is frozen in `arch_s0s1/freeze.json`.

**Files:**
- Code: `reft/arch_s0s1_adapters.py`, `reft/arch_s0s1_train.py`, `reft/arch_s0s1_evaluate.py`.
- Configuration: `configs/reft/arch_s0s1_frozen.yaml`.
- Tests: `tests/reft/test_arch_s0s1.py`.
- Slurm: `slurm/reft_arch_s0s1*.sbatch` and `slurm/reft_arch_s0s1_submit.sh`.
- Rules: `ARCH_S0S1_DECISION_RULES.md`.

**Type.** A bounded architecture-screening experiment. The exposed dev150 panel both selects checkpoints and evaluates
them; this is not confirmation.

**Out of scope:** E1+D2, P2-3, P3, LoRA, and any rank, layer, seed, loss-weight, optimizer or data sweep. There are no
additional data evaluations.

## 1. Question

Does explicitly separating the decoder residual and the cross-attention contribution in a low-rank representation
update improve correction–preservation behavior beyond a parameter-matched increase in standard LoReFT rank?

| Hypothesis | Meaning |
|---|---|
| H1 | Standard LoReFT is limited by rank/capacity |
| H2 | Reading the cross-attention contribution separately is a useful interface |
| H3 | Neither change solves the correction–preservation bottleneck |
| H4 | Results depend substantially on initialization, optimization or exposed-dev variation |

**Claim boundary.** M1 is *cross-attention-conditioned*, not acoustically grounded. u is query-dependent and already
contained in h = q + u. An M1 advantage would not show acoustic dependence or causality. R3 and M1 match parameter
counts, not representational structure: R3 has a rank-3 output subspace, M1 a rank-2 output subspace with a richer
input readout. The comparison is between two parameter-matched architectures, not an isolated acoustic-information
effect.

## 2. Site (unchanged R2 semantics)

Decoder layer 31 of the frozen Whisper-large-v3 (BF16, eager attention), post-cross-attention residual, pre-FFN. In
transformers 4.57 `WhisperDecoderLayer.forward`:

```
residual = hidden_states                                   # q
s = encoder_attn(encoder_attn_layer_norm(residual), H_enc) # u (dropout is the identity in eval mode)
h = residual + s                                           # h = q + u -> final_layer_norm, FFN and residual
```

Both new adapters reuse the R2 hooks (`reft/adapters.py`):
- `q` comes from the `encoder_attn_layer_norm` pre-hook input;
- `u` is the `encoder_attn` forward-hook output of the same call;
- `h = q + u` is formed in the backbone dtype, exactly as R2 does;
- the update δ is computed in float32 and returned as `u + δ.to(u.dtype)`, so h̃ = h + δ reaches the FFN branch and the
  residual path;
- absolute positions come from the layer's `cache_position`, and δ applies only at positions ≥ `start_pos` = 3;
- `disabled()` bypasses the adapter for the frozen Auto-LID decision;
- hooks are removed on exit, and M1 consumes and clears `q` at each call, so no stale residual is reused.

M1's u is the actual cross-attention output at that site. It is not the encoder states, attention weights or an
acoustic embedding, and it is never detached or overwritten before the readout.

## 3. Architectures

| Arm | Update | Trainable tensors | Parameters |
|---|---|---|---|
| R2 (historical reference; P2-1b `KL_seed{42,43,44}`, never retrained) | h + Rᵀ(W h + b − R h) | R 2×1280, W 2×1280, b 2 | 5,122 |
| **R3** | h + R₃ᵀ(W₃ h + b₃ − R₃ h) | R₃ 3×1280, W₃ 3×1280, b₃ 3 | **7,683** |
| **M1** | h + Rᵀ(W_h h + W_u u + b − R h) | R 2×1280, W_h 2×1280, W_u 2×1280, b 2 | **7,682** |

**R rows** are orthonormal through torch's orthogonal parametrization, exactly as in R2. The trainable tensor is the
parametrization's `original`; the `base` buffer is not a parameter. Counts come from the actual trainable tensors.

**Initialization:**
- **R3:** `reft.adapters.LoReFTAdapter(r = 3)`, unchanged code. R₃ is the QR of a seed-s Gaussian (1280 × 3); W₃ = R₃;
  b₃ = 0.
- **M1:** subclasses the R2 intervention. R, W_h (the R2 field `learned.weight`) and b (`learned.bias`) are built by the
  R2 constructor itself, so under the same global seed M1's initial R and `base` equal R2's. W_u = 0 is created without
  consuming RNG.

Both give δ = 0 exactly at initialization. There is no gate, scaling network, layer norm or trainable α.

**M1 → R2 reduction.** With W_u = 0, the branch W_u u adds exact IEEE zeros. M1 with R2's R, W and b therefore
reproduces R2's update and logits bitwise; G3 tests this.

## 4. Frozen training protocol (both arms identical)

| Setting | Value |
|---|---|
| Data | the frozen 500 training utterances (`p2_0/train_manifest.json`); order `numpy default_rng(s + epoch).permutation` |
| Seeds | 42, 43, 44. `torch.manual_seed(s)` → backbone → adapter (P2-1b order) |
| Schedule | 3 epochs; 63 micro-batches of 8, accumulation 2 with the final partial step → 32 updates/epoch, 96 per run |
| Optimizer | AdamW, lr 5e-4, betas (0.9, 0.99), ε 1e-6, weight decay 0, clip 1.0, constant schedule |
| Precision | backbone BF16 (frozen); adapter parameters and update FP32 |
| Objective | `reft.p2_1b_train.kl_step`, unchanged: token-mean CE + 1.0 · KL(p₀ ‖ p_θ), full-vocabulary FP32, teacher detached, mask = label ≠ −100 ∧ teacher argmax = label (content tokens and EOS) |
| Decoding | frozen Auto-LID (adapter disabled for LID), beam-5 primary, greedy secondary, canonical CUDA 12.8 environment |

There is no on-policy event builder, no reference-fallback loss and no extra regularizer.

**Training loop.** `goldkl_epoch` mirrors `p2_1b_train.train_loop` (arm KL) operation for operation. It only adds
read-only diagnostics per optimizer step:
- per-tensor pre-clip gradient norms;
- update norms and parameter norms (W_u included for M1);
- the orthogonality error max|RRᵀ − I|;
- per-micro-batch CE, GoldKL and mask counts.

G6 proves the loop reproduces the historical R2 KL_seed42 epoch-1 adapter bitwise.

**Pairing.** Architectures have differently shaped tensors, so initializations cannot be equal across arms. The
requirement is deterministic matched-seed procedures:
- **Hashes:** every initial tensor, including `base`, is hashed per method and seed.
- **M1:** its `base` must equal R2-GoldKL seed s (training aborts otherwise).
- **Per epoch:** supervised-token counts, GoldKL mask positions and update counts must equal the matched R2-GoldKL run.
  The mask depends only on the frozen teacher and the batch order.

**Every run writes, write-once under `arch_s0s1/training/{R3,M1}_seed{s}/`:**
- epoch-0 zero-init dev150 beam-5 identity (150/150, else abort) and greedy;
- per-epoch CE, GoldKL, total loss, gradient norms and step diagnostics;
- the adapter after epochs 1–3;
- beam-5 and greedy dev decodes per epoch, with a frozen-LID check;
- parameter counts, runtime, memory, input and checkpoint hashes, and a weights check.

## 5. S0 engineering preflight (one GPU job, no S1 training; gates are exact as implemented in `run_preflight`)

| Gate | Requirement |
|---|---|
| **G1** provenance | branch `cs-asr-reft`; this freeze verified, including the P2-0 → P2-1b (via P2-2), P2-1c and P2-2 chains; R2-GoldKL adapters match their recorded hashes; pinned B0; frozen manifests; data audit valid with no locked dialogues; expected weights; canonical CUDA; no S1 outputs yet. Git head, dirty paths and the queue are recorded |
| **G2** parameters / identity | R2 5,122, R3 7,683, M1 7,682 trainable; orthogonality ≤ 1e-5; W = R, b = 0 (M1 also W_u = 0); module output exactly 0 at init; teacher-forced logits bitwise equal the frozen model; perturbed adapter under `disabled()` bitwise frozen; positions 0–2 unchanged and content positions changed (start_pos = 3); zero-init beam-5 and greedy equal the pinned baseline on 20 dev utterances (R3, M1); M1's initial shared tensors (incl. base) equal R2 seed 42 |
| **G3** M1 ≡ R2 | Zero-init R2 vs M1: shared tensors and logits bitwise. A perturbed R2 and the trained R2-GoldKL seed-42 selected checkpoint, transplanted into M1 with W_u = 0: parameter sets differ by exactly `w_u`; shared tensors, the update and teacher-forced logits are bitwise equal (exact tolerance, differences reported); cached greedy absolute positions identical; transplanted M1 re-decodes the recorded R2 beam-5 and greedy dev outputs 20/20 |
| **G4** gradients | Trainable tensors exactly the intended set; finite loss; backbone has no gradients; at init gradients reach W and b (and W_u for M1). Perturbed: GoldKL > 0, gradients nonzero and finite on every tensor including R. Hooked logits equal the frozen forward with the manually recomputed update added at the Site (single application at the exact site) on two different batches, with exactly one hook call per forward (no stale u). A bounded 5-step AdamW smoke on one training micro-batch reduces the loss, moves every tensor, keeps R gradient nonzero after leaving identity and orthogonality ≤ 1e-5. Hooks cleared |
| **G5** decoding | Perturbed R3/M1: frozen LID unchanged with the adapter installed (20/20); greedy and beam-5 cache positions start at [0, 1, 2, 3] and increase by one; cached greedy vs teacher-forced argmax under canonical suppression ≤ 1 mismatch per utterance (2 utterances); the perturbed adapter changes at least one of the 20 dev beam-5 decodes, and X–Y–X no state leakage and repeat determinism are checked on such an utterance; hooks cleared. **FP32 semantic check** (P2-1c amendment r1): with the model cast to float32 in memory, incremental cached vs full forward max |Δ log p| ≤ 1e-3, argmax agreement where the margin ≥ 0.2, and correct positions on 3 dev utterances. The BF16 values and beam-score-vs-teacher-forced differences are recorded descriptively and are not gated |
| **G6** GoldKL parity | At zero init (R2, R3, M1): teacher = frozen forward; student = teacher bitwise; initial KL ≤ 1e-6; CE equals the P2-1 CE path; gradients equal the CE-path gradients within (rtol 1e-4, atol 1e-7); the label contract holds; the teacher-correct mask, initial CE and initial KL are identical across the three architectures. **Regression:** the shared S1 loop configured with historical R2 (seed 42, 1 epoch) reproduces `p2_1b/training/KL_seed42/epoch1/adapter.pt` bitwise, with an equal init `base` |
| **G7** resources / freeze | Micro-batch forward+backward timing and teacher overhead for R2/R3/M1; peak allocated/reserved VRAM; adapter file bytes; parameter + Adam-state bytes; decode latency (zero-init and perturbed); projected job ≤ 9,000 s (3 h limit); reserved < 38 GB; authorization present; weights unchanged; freeze verified |

All pass → `ARCH_S0_READY_FOR_S1`. Otherwise → `ARCH_S0_BLOCKED_<gates>`: stop, with no silent modification of the
architecture, hyperparameters or freeze rules. The run refuses to start unless the newest non-dry preflight is READY.

**Pre-freeze dry runs.** `ARCH_S0S1_DRY_ROOT` redirects outputs to scratch. Dry runs exercise only the preflight code
(no S1 run is possible) and are deleted after use. G1 cannot pass in a dry run by design.

## 6. S1 runs and chain

Exactly six new runs: R3_seed42/43/44 and M1_seed42/43/44 (`slurm/reft_arch_s0s1_submit.sh`, `afterok`, kill on an
invalid dependency, at most two concurrent GPU jobs):

```
train ×6 → select → evaluate ──┐
         ├→ verify ─────────────┼→ report
         └→ mech-anchors → mech-gpu ┘
```

**Selection** uses the unchanged P2-1 rule per run: lowest dev150 beam-5 genuine-CS MER, then MER, then Mandarin CER,
then the earliest epoch. Selection never uses retention, switch-back counts, English net, the P2-1c cases, mechanism
results or training loss. All epochs are reported.

**Verify:** every epoch adapter's hash, reload, parameter count, and re-decode of the first 20 dev utterances
(beam-5) as recorded.

## 7. Evaluation (CPU; canonical scorer and P2-1b / P2-1D / P2-2 metric code)

**Systems:** B0, R2-GoldKL (recomputed; must reproduce the P2-1b evaluation exactly), R3 and M1. Every epoch, beam-5
(primary) and greedy (secondary).

**Metrics:**
- **Recognition:** MER, CS MER, English WER, PIER, Mandarin CER.
- **English:** substantive corrections, corruptions and net; English fates (omission / translation-like / lexical
  substitution → correct, correct → wrong, wrong → other wrong).
- **Mandarin:** new errors, repairs and retention.
- **Damage and stability:** outside-target damage; loops and non-termination; improved and worsened utterances and
  dialogues.
- **P2-1D failure metrics:** new deletions, long dropped runs, long switch-back dropped units, damage within / beyond 7
  units of a switch, damage in utterances without an English repair, worst-dialogue regression, switch-back span fates.

**Comparisons.** Paired M1 − R3 (primary), M1 − R2 and R3 − R2, per seed and as the seed mean. Each comparison reports:
- dialogue-cluster bootstrap (2,000 draws, seed 42, the same draws for every arm);
- CS-only Mandarin and switch-back counts;
- worst-dialogue contribution and leave-one-dialogue-out range;
- epoch-matched secondary differences.

The 20 dialogues are shared across seeds; intervals are exploratory and descriptive.

## 8. Bounded mechanism checks (after training and the primary evaluation; outside the verdict)

**13.1 contribution audit** (GPU, reference-free). For the selected R2/R3/M1 checkpoints, the system's own dev150
beam-5 outputs are teacher-forced. At positions ≥ 3, the job records:
- the pre-projection edit e = (W_h − R) h + W_u u + b and its parts e_h, e_u, b;
- ‖W_h h‖, ‖R h‖ and ‖u‖;
- the post-projection update eR. Rows are orthonormal, so ‖eR‖ = ‖e‖; this is checked;
- ‖update‖ / ‖h‖;
- cos(e_h, e_u), cos(e_u, e), the share of e_u in the squared edit, and the fraction of positions with ‖e_u‖ > ‖e_h‖.

These are summarized overall and by the predicted token's script (Han → ZH, Latin → EN). They are descriptive, not
causal.

**13.2 prefix-history cases.** The P2-1c replay engine runs on the ten fixed P2-1c cases for B0, R2-GoldKL 42–44 and the
selected R3/M1 checkpoints, under the same conditions as P2-2: original; B0 `pre_en`, `post_en` and `early_zh`
prefixes; own prefix; B0 at own prefix; local next-token Δ; sequence preference. The R2 rows must reproduce P2-1c. These
cases are exploratory and reference-informed; they never choose checkpoints, alter S1 criteria or tune M1.

There are no acoustic swaps, encoder patching, phoneme models or new counterfactual controls.

## 9. Claim boundaries

Not established:
- generalization (one exposed panel, 20 dialogues);
- any acoustic or causal role of u;
- superiority over an optimized LoRA;
- results under any other rank, layer, objective weight or optimizer.

Identical parameter counts do not imply identical training compute; compute is reported.
