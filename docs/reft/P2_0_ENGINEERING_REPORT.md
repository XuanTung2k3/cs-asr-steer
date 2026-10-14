# P2-0 — engineering preflight report

**Verdict: `P2_0_BLOCKED_G2_BASELINE_ENVIRONMENT_DRIFT`.**

G1 and G3–G7 pass. G2 fails on one frozen requirement: the two-stage decode must reproduce the **sealed P0 outputs**.
- The decoding policy itself is equivalent. Native `generate(language=None)` equals the two-stage frozen-LID decode on
  150/150 dev utterances.
- But the P2 environment reproduces the sealed P0 tokens on only **133/150**.
- A controlled diagnostic shows that the 17 differences are caused entirely by the cuBLASLt library version:
  - P0 ran with the env's CUDA 12.1 `libcublasLt` shadowing torch's 12.8 (the P1 amendment r4 defect);
  - P2 must load the correct 12.8 libraries.
- The gate is not lowered. The minimal repair below needs project-owner approval.

**P2-1 training was not run.** Its launcher refuses to start without the owner-created `p2_1/AUTHORIZATION.json`.

**Provenance.**
- Protocol `P2_0_PROTOCOL.md`; commit `f4bc3b7`; freeze `freeze/p2_0.json` (sha256 `1574947e…`).
- Slurm jobs:
  - CPU tests 59206;
  - data audit 59207;
  - GPU smoke 59208;
  - G2 library diagnostic 59212;
  - baseline scoring 59213.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| G1 data and exposure | **PASS** | 500 training utterances (27 dialogues; 391 former P1 construction and 109 former P1 audit, now training data; disclosed) vs 150 dev (20 dialogues). 0 shared dialogue, conversation, speaker, corpus speaker, audio hash or utterance ID. No locked-role dialogue. Locked roles never opened. Manifests `db4a3d26…` (train) and `97f4d24e…` (dev). Segment-time overlap cannot be checked (no timestamps). |
| G2 Auto-LID equivalence | **FAIL** (environmental) | LID (stage A) equals P0 and native on 150/150. Native = two-stage: beam-5 150/150, greedy 20/20. Training LIDs recomputed equal 500/500. LID unchanged with either adapter installed: 20/20. Both methods use the same prompt policy. **Two-stage = sealed P0 tokens: 133/150** (termination 149/150, text hash 134/150). |
| G3 teacher-forcing compatibility | **PASS** | CPU tests: shifting, labels at 3 + t, prefix and padding masked, EOS supervised once, invalid targets rejected, and loss independent of the prefix and padding logits. Content token 0 is trainable only if the adapter covers position 3. Target audit: every target ends in one EOS with ids < EOS, max 127 tokens; 1 of 18,116 content tokens is in the generation suppress list (documented). The training prompt is the frozen Auto-LID prompt. Identical objective for R2 and L1. |
| G4 adapter identity and gradient flow | **PASS** | R2 5,122 / L1 5,120 trainable parameters (exact); 0 backbone parameters trainable. Zero-update teacher-forced logits are bitwise equal. One AdamW step: finite gradients on every LoReFT tensor; LoRA gradients on B, with A zero at step 0 as expected while B = 0 and nonzero after the step. Effective update nonzero (max logit change 0.59 R2, 0.125 L1). Frozen logits and weights unchanged. R stays orthonormal (1.2e-7). |
| G5 free-decoding correctness | **PASS** | Zero-update beam-5 tokens equal the frozen baseline on 150/150 (both methods); greedy 20/20. Absolute positions seen by the hook: prefill [0, 1, 2, 3], then +1 per step (greedy and beam-5). With random nonzero adapters, cached greedy equals teacher-forced argmax with 0 mismatches (1-shifted alignment: 18/19 mismatches). R2 leaves prefix logits 0–2 bitwise unchanged, and an update from position p leaves logits before p bitwise unchanged. Beam-5 transition scores vs teacher-forced rescoring: max 0.054 nats, mean ≤ 0.012 (tolerance 0.15 / 0.03). Decoding X, Y, X leaks no state. Batch vs batch-1 teacher-forced argmax agreement 1.000. No hooks left; LoRA targets restored. |
| G6 compute | **PASS** | pip cuBLAS/cuBLASLt 12.8 consistent. Weights unchanged. Peak 6.0 GB on an H100 MIG 3g.40gb slice. Micro-batch of 8 (forward and backward): 0.37 s. Projected P2-1: about 25 minutes per method (3 epochs plus 4 dev decodes), far below 3 h. Smoke wall time 16.1 min. |
| G7 reproducibility | **PASS** | Config and protocol frozen before the smoke. CPU tests 44/44 (11 new) and the GPU-job test run 11/11. Hashes recorded. Adapter save/load bitwise. Safe commit and push (below). |

## G2 diagnosis (Slurm 59212)

The test took the 17 P0-mismatching rows plus 5 matching control rows and decoded them natively and two-stage under each
library resolution.

| Process environment | Mapped `libcublasLt` | Native = two-stage | 17 mismatch rows = sealed P0 | 17 rows = P2 baseline | Controls = P0 |
|---|---|---:|---:|---:|---:|
| P0 launcher (env `lib/` first) | 12.1.0.26 (shadowing) | 22/22 | **17/17** | 0/17 | 5/5 |
| P2 launcher (pip libs first) | 12.8 (pip) | 22/22 | 0/17 | **17/17** | 5/5 |

Conclusions:
1. The decoding contract is exact in both environments.
2. The change against P0 is a BF16 kernel difference between cuBLASLt 12.1 and 12.8 that flips near-tie beam decisions.

**Effect on the frozen baseline** (scored with the P0 evaluator):

| Metric | Sealed P0 (cuBLASLt 12.1) | P2 environment (12.8) |
|---|---:|---:|
| Overall MER | 0.1178 | 0.1199 |
| Genuine-CS MER | 0.1206 | 0.1227 |
| English WER / PIER | 0.205 / 0.191 | 0.211 / 0.197 |
| Mandarin CER | 0.095 | 0.096 |
| Substantive English errors | 215 | 222 |

The P2-environment greedy baseline, which is new, has MER 0.149 and CS MER 0.137.

**Why not run P2 in the P0 environment.** That library mix produced the `cublasSdot` SIGFPE and `cublasSgemm`
`NOT_INITIALIZED` failures in P1 (amendments r3 and r4). P2 trains float32 adapters through it. Training on a
known-defective mix to reproduce a historical artifact would trade a reproducible, documented baseline shift for
uncontrolled numerical failures.

## Minimal repair (owner decision required; not applied)

Approve a timestamped P2-0 amendment that **re-anchors the P2 frozen baseline to the corrected CUDA environment**:
- The P2 baseline becomes `p2_0/frozen_baseline_dev150_{beam5,greedy}.jsonl` (sha256 `734f02e2…` and `e689a2f5…`).
  This is the two-stage frozen-LID decode, equal to native Auto on 150/150 in the same environment.
- The P0 sealed outputs remain the historical P0 record.
- G2's reproduction target becomes "two-stage = native frozen Auto in the P2 environment (150/150), with the P0 drift
  documented (17/150 utterances, ΔMER +0.0021)".
- Every P2-1 comparison (R2, L1 and the baseline) runs in the same corrected environment.

No code change is needed. With that amendment, all seven gates pass on the existing evidence, and the status would be
`READY_FOR_P2_1_TRAINING`.

## Commands prepared (NOT executed)

After approval and creation of `/mnt/data/tungnx/cs-asr-reft/p2_1/AUTHORIZATION.json`:

```bash
sbatch slurm/reft_p2.sbatch train R2     # LoReFT rank 2, lr 5e-4, 3 epochs, ~25 min, ~6-8 GB
sbatch slurm/reft_p2.sbatch train L1     # LoRA q/v rank 1, lr 1e-4, 3 epochs, ~25 min, ~6-8 GB
sbatch slurm/reft_p2_cpu.sbatch select /mnt/data/tungnx/cs-asr-reft/p2_1/R2
sbatch slurm/reft_p2_cpu.sbatch select /mnt/data/tungnx/cs-asr-reft/p2_1/L1
sbatch slurm/reft_p2_cpu.sbatch evaluate /mnt/data/tungnx/cs-asr-reft/p2_1/R2 /mnt/data/tungnx/cs-asr-reft/p2_1/L1
```

Each training job:
- re-checks the freeze and manifest hashes;
- reproduces the frozen baseline with the zero-initialized adapter (150/150);
- trains for 96 updates, saving the adapter each epoch;
- decodes dev150 each epoch (beam-5 and greedy, frozen LID, LID re-checked with the adapter disabled).

## Compute used by P2-0

| Job | Purpose | Wall time |
|---|---|---:|
| 59208 | GPU smoke | 16:07 |
| 59212 | G2 diagnostic | 2:16 |
| CPU jobs | | under 1 min total |

No optimizer step beyond one per method in the smoke. No P2-1 training.
