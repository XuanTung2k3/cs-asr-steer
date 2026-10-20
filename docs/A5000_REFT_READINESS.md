# A5000 ReFT readiness

Audit date: 2026-10-20. Published source: `cs-asr-reft` at
`000b0629e55ea2e6b65812e829a085143f9bad2a`. No GitHub push.

## Verified source and proposal

The GitHub branch still has the previously observed HEAD. Its actual source is more extensive than
README_REFT's older P0/P1 description, but it does **not** contain the latest H100 encoder-output or P3 work.
There is no `E-ReFT` encoder-output adapter or same-site LRA in this checkout. `reft/adapters.py` provides
decoder-layer-31 post-cross-attention, pre-FFN LoReFT and cross-attention weight-LoRA.
`reft/arch_s0s1_adapters.py` adds the decoder conditional M1 control. These are different intervention sites
and implementations from the proposed E1 encoder-output experiment.

Reviewed local proposal: `CS_ASR_ReFT_Research_Proposal_Workflow_2026-10-19.md`, heading
“Research proposal v5 — Compact Representation Adaptation for CS-ASR”, updated 2026-10-20.
Its SHA-256 is `ff580a9f318560f2db760ca3c3b6c3b903d2c136e0638a2a3467e717d665b711`.
Local path: `/Users/xntung/Works/VinUni/Research/ASR_CS Multilingual/ACL Draft/`.
No ReFT proposal v5.1 was found. The unrelated older `Downloads/CS-ASR_proposal_v5.md` concerns the earlier
language-steering proposal and is not the new objective experiment specification.

Local inventory covered Code, Works, Downloads, Documents, Desktop and the old `cs-asr-steer-stage` tree;
no newer ReFT checkout, worktree or Git bundle was found. The stage tree contains older V2R3 hypothesis
substitution code; the local `cs_asr_pilot.zip` is an earlier probing project. Neither supplies missing
encoder ReFT, historical checkpoints or P3 controllers. `Documents/H100 connect` is an empty Git repository.
No H100 connection was attempted.

## Compatibility matrix

| Capability / experiment | Available implementation | A5000 readiness / limitation |
|---|---|---|
| CUDA / Whisper-large-v3 | Canonical repository loader and preprocessing | G0 validated; details in setup report |
| E0 official training inventory | `scripts/a5000/validate.py e0`, existing `csasr.data.manifest` | Portable training-only audit; no reconstruction of historical roles |
| E0 CE shift and masks | `reft.auto_policy.build_tf_batch`, `tf_loss` | CPU tested: prompt positions 0–2 and padding ignored; content+EOS token mean |
| E0 full objective/label specification | v5 proposal, some reusable text/script helpers | Not complete: O2/O3 alignment, ambiguity/UNKNOWN policy, auxiliary site and logical-batch reduction still need an explicit new protocol and implementation |
| Available decoder LoReFT CE | `reft.adapters.LoReFTAdapter`, `reft.train_supervised.batch_loss` | Engineering smoke only; historical training runner needs unavailable frozen panels/seals |
| Decoder GoldKL bridge | `reft.p2_1b_train.kl_step` | Code and CPU tests present; full runner blocked by historical inputs and source/environment checks |
| E1 encoder-output E-ReFT, r=2 | Missing from published branch and verified local backups | Missing implementation; decoder success does not establish encoder training memory feasibility |
| E1 encoder-output same-site LRA | Missing | Missing implementation; do not substitute decoder weight-LoRA |
| E1 practical LoRA comparison | Decoder cross-attention LoRA primitive exists | Proposed strong practical objective-matched comparator is not restored |
| O2 auxiliary token LID CE | Missing proposed objective/head and audited gradient path | Requires scientific implementation; not added during setup |
| O3 normalized weighted CE | Missing proposed objective | Existing region-weighted **KL** is a different objective; not relabeled weighted CE |
| New D1 language/lexical decomposition | Some reusable text/token-language utilities | Full v5 diagnostic runner absent; requires encoder arm/checkpoints and declared vocabulary partitions |
| New D2 controlled readout | Legacy probing utilities exist | Full v5 fixed/refit, prefix-controlled package absent; requires labels, arms and checkpoints |
| Historical D1 region-KL / D2 edit-locality | Source/configs/tests present | Names refer to older experiments; artifact-bound runners blocked, historical configs remain frozen |
| P2-2 continuation / ARCH-S0/S1 | Source/configs/tests present | Full historical runs require missing private manifests, baselines, checkpoints and freeze chains |
| Later P2-F1 | Missing | Newer unpushed source/artifacts required |
| P3-S / P3-M controller workflow | Missing | Newer unpushed source/artifacts required; not invented |
| Historical train394 / inner106 / full500 | Missing | Obtain exact manifests and hashes from a verified backup; no random recreation |
| Historical comparisons / H100 parity | Model identity anchor available | Decoding/environment/source/panel parity unestablished; no historical parity claim |

## Scientific boundaries and next work

Official training audio and original split indexes are sufficient for infrastructure checks. They do not
reconstruct the role assignment (`loc-train`, `util-train`, router/calibration/confirmation locks), training
panels, selected adapter checkpoints, ACL-0 seals or freeze chains used by historical experiments.
Recover those exact files and the later source from an authenticated backup before attempting historical
reproduction. Do not treat an A5000 engineering record as historical authorization.

E0 can inventory data and validate the existing decoder CE conventions now. To complete E0/E1 for v5,
recover or separately implement and review encoder-output E-ReFT/LRA, declare the exact training and
selection manifests, settle objective/label/reduction specifications, and then run the intended encoder
backward-memory preflight. That scientific implementation is beyond this setup task. O2/O3, D1/D2 and
full objective-matrix training remain unexecuted.

The next runnable E0 command is:

```bash
ssh A5000 'cd ~/tung.nx/cs-asr-reft && scripts/a5000/run.sh e0'
```

The bounded smoke of the **available decoder** implementation is:

```bash
ssh A5000 'cd ~/tung.nx/cs-asr-reft && scripts/a5000/run.sh smoke'
```

There is no runnable encoder-output E1 command in this source version. No multi-epoch or full-matrix
training has been started. Consult `A5000_SETUP_REPORT.md` for actual validation outcomes and memory.
