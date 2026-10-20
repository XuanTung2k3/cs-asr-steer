# A5000 setup and validation report

Status: **`A5000_BASE_ENVIRONMENT_READY_REFT_CODE_INCOMPLETE`**

Completed 2026-10-20. The available published decoder ReFT implementation runs on A5000. The newer
encoder-output E-ReFT/LRA, P2-F1 and P3-S/P3-M source and historical artifacts were not found, so the full
research workflow and proposed E1 are not declared restored.

## Deployment and preservation

| Item | Verified value |
|---|---|
| Local repository | `/Users/xntung/Code/cs-asr-reft` |
| Remote repository | `/home/ubuntu/tung.nx/cs-asr-reft` |
| Git branch | `cs-asr-reft` |
| GitHub, local and remote HEAD | `000b0629e55ea2e6b65812e829a085143f9bad2a` |
| Source | `https://github.com/XuanTung2k3/cs-asr-steer.git` |
| Active Python | `/home/ubuntu/tung.nx/reft-a5000/bin/python` |
| Validation output root | `/home/ubuntu/shared/tung.nx/a5000-validation` |

The destination did not exist before transfer. SSH, permissions, filesystem capacity and existing directories
were inspected first. All 620 tracked files and Git metadata were transferred using rsync over SSH, with
executable bits preserved. Repeat deployment uses `scripts/a5000/deploy.sh`, checks matching HEAD and no
tracked remote edits, uses `--update`, excludes caches/environments/secrets/large local artifacts and never
uses `--delete`. Repeated transfer was exercised. Original tracked scientific code/configs/evidence remain
unchanged; new A5000 files are additive, uncommitted working-tree files. No GitHub push or local commit.

No H100 access was used. No newer local ReFT worktree or bundle was found in the inspected work/backup roots.
The local older stage and pilot ZIP do not contain the missing methods. The reviewed ReFT v5 proposal and
source/backup distinctions are documented in `A5000_REFT_READINESS.md`.

## Runtime and artifacts

The host has six RTX A5000 GPUs (24,564 MiB each, compute capability 8.6), driver **535.309.01**. One idle GPU
was used directly; no scheduler commands were exposed. Python **3.13.3**, PyTorch **2.7.1+cu126**, CUDA wheel
runtime **12.6**, Transformers **4.57.6**. The driver reports CUDA **12.2**. Actual CUDA operations and the
bounded model workload succeeded. CPU/RAM/disk details, all package pins and environment reproduction are in
`A5000_ENVIRONMENT.md`; full `pip freeze`, inherited ASR freeze and system/GPU metadata are in the external
`environment/` output directory. `pip check` passes. The existing `asr` environment was not modified.

Verified [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1):

```text
/home/ubuntu/shared/tung.nx/models/whisper-large-v3/06f233fe06e710322aca913c1bc4249a0d71fce1
```

All 11 required model/processor/config files match authoritative metadata at that revision. The weight file
SHA-256 is `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`; the BF16 parameter hash is
`0b2acef462086d27b205b951e2f5c0964bd75d671e0c68b91c58a215b75db055`, matching the repository model anchor.
The existing cache's tokenizer config and special-token map had altered pad-token settings. It was preserved;
a separate pinned model view contains authoritative small files and symlinks the verified existing weight.
Model identity does not establish historical H100 transcript parity.

Verified [CS-Dialogue release](https://huggingface.co/datasets/BAAI/CS-Dialogue/tree/a84770fcdf14072252373a0227321227eaac7dc7):

```text
/home/ubuntu/shared/tung.nx/CS-Dialogue
revision: a84770fcdf14072252373a0227321227eaac7dc7
training audio root used by reader: data/short_wav/extracted
original index root: data/index/short_wav
```

All 19 short_wav archive parts match authoritative LFS size and SHA-256. Total local dataset usage is about
17 GiB, including 9.0 GiB compressed and 7.5 GiB extracted training audio. Original membership:

| Official split | Utterances | Speakers | Setup access |
|---|---:|---:|---|
| train | 26,239 | 140 | Audio headers/hashes and training text; one bounded smoke utterance |
| dev | 6,186 | 30 | ID/path inventory only |
| test | 6,257 | 30 | ID/path inventory only |

IDs and speakers are disjoint across official splits. All training audio headers are readable, 16 kHz,
totaling **68.64181 hours**. The existing repository reader built the training-only manifest and verified the
dialogue-pairing structure. No dev/test transcript was downloaded and no dev/test audio was extracted,
decoded or scored. Long-form split indexes and speaker metadata were inventoried; long-form audio was not
downloaded. Exact historical research-role assignments, train394/inner106/full500, panels and checkpoints
are unavailable and were not recreated.

Training manifest:

```text
/home/ubuntu/shared/tung.nx/a5000-validation/e0/official_train.parquet
SHA-256: e6e0003ce32ade76e163b7e956a7ca737785de21bcafdf04330163c133aa1c64
```

E0 reruns preserved this manifest hash. This is an official-training inventory, not a historical split or a
completed v5 objective/label specification.

## CPU validation

**989 distinct tests accounted for: 988 passed, zero outstanding failures/errors, one skipped/blocked
artifact-pinning test.** This is the union of actual full-suite and focused regression results, not a claim
that the original raw full-suite run had zero failures:

| Run | Passed | Failed | Skipped | Notes |
|---|---:|---:|---:|---|
| Full published suite, final runtime | 985 | 1 | 3 | 499.71 seconds; missing uroman at the time this test ran |
| ReFT-only suite | 156 | 0 | 2 | Includes construction, hooks, CE masks/shift, gradients, serialization, determinism |
| Alignment diagnostic after installing uroman | 37 | 0 | 0 | Entire affected module passed |
| Last failed test after full run completed | 1 | 0 | 0 | Confirms the raw-suite failure is repaired |
| Two existing real-tokenizer tests at portable model path | 2 | 0 | 0 | Runtime-only cfg lookup override; frozen files unmodified |

The remaining skip is `tests.test_lss_dialogue_roles::test_current_configuration_reproduces_the_accepted_v2_allocation`:
the historical published dialogue manifest is absent. This historical artifact check is blocked, not supplied
with replacement data. The suite's empty-stratum and pandas compatibility warnings remain visible in raw logs.
There are no manufactured private panels, checkpoints or approvals. No scientific test expectation was weakened.

Evidence: external `cpu/summary.json`, `cpu-final.xml`, `cpu-uroman.xml`, `cpu-repaired.xml`, `tokenizer.xml`.
The summary deduplicates test identities and records each overriding regression result explicitly.

## Bounded GPU validation

Two complete final runs took **38.26 s** and **37.92 s**, including model loading/hashing. Each was capped at
600 seconds; no multi-epoch training, tuning sweep or full objective matrix was launched.

| Check | Actual result | Peak allocated GiB | Peak reserved GiB |
|---|---|---:|---:|
| G0 | FP16/BF16 CUDA, canonical large-v3 BF16 load, 32+32 layers, 128 mel bins | 2.884 | 3.125 |
| G1 | One official-training CS utterance, canonical Auto-LID beam 5, cap 200, EOS reached, repeat tokens identical | 4.087 | 4.477 |
| G2 available decoder ReFT | Rank 2, decoder 31 post-cross-attention/pre-FFN, 5,122 trainable scalars, microbatch 1, one optimizer step plus a second backward-only gradient check | 3.173 | 3.441 |
| G2 intended encoder-output ReFT | **Not run: implementation missing** | — | — |
| G3 | Training manifest → real audio → canonical target preparation → inference/CE → canonical MER export → private output files | Covered above | Covered above |

Decoder CE was finite: **0.4196994** before the step, **0.4148011** afterwards. All authorized adapter gradients
were finite, intended parameters moved, backbone parameters were frozen without gradients, and the complete
backbone parameter hash stayed unchanged. Checkpoint save/reload preserved exact state and logits. Exact
no-op initialization and disabled-adapter identity passed; no new scalar-strength interface was invented.
No activation checkpointing or gradient accumulation was needed. The actual serialized adapter is
**6,576,349 bytes**, including orthogonal-parametrization buffers; 5,122 trainable scalars do not imply a 20 KB file.

The rerun produced the same loss, checkpoint SHA-256 and predictions in a new directory; the previous completed
report's SHA-256 remained unchanged. Both completed records remain available:

```text
smoke/20261020T112237169607Z/report.json
smoke/20261020T112352238100Z/report.json
```

These paths are under the external validation root. Predictions, training-side scores and checkpoints remain
outside Git. Earlier failed diagnostic attempts were preserved: an old Transformers `dtype` API failure,
a model-cache metadata mismatch and a test-scaffold JSON serialization failure. The latter was repaired by
converting canonical alignment dataclasses to JSON records; MER computation was unchanged.

The measured workload fits one A5000. It does not establish full-length/batch training capacity, encoder-output
gradient memory, the complete v5 objective matrix, or historical numerical/transcript parity.

## Added files and next command

All portability work is separate from frozen research source:

- `configs/a5000/runtime.yaml`: pinned roots/model identity and actual available decoder site.
- `scripts/a5000/run.sh`, `validate.py`: standalone training-only audit and bounded smoke.
- `scripts/a5000/download_data.py`: resumable exact-release download and authoritative hashes.
- `scripts/a5000/prepare_model.py`, `verify_model.py`: preserved-cache model view and authoritative file checks.
- `scripts/a5000/setup_environment.sh`, `capture_environment.py`: environment-local overrides and full locks.
- `scripts/a5000/cpu_artifacts.py`, `summarize_cpu.py`: existing artifact tests and explicit JUnit aggregation.
- `scripts/a5000/deploy.sh`: repeatable SSH/rsync deployment.
- This report, `A5000_ENVIRONMENT.md`, `A5000_REFT_READINESS.md`.

Runnable now: E0 training-side inventory, canonical inference, available decoder LoReFT CE smoke and CPU
primitive tests. Historical CE/GoldKL/continuation/architecture runners require their exact missing artifacts.
New encoder-output E1, O2/O3, proposed D1/D2 and P2-F1/P3 remain unavailable; see the readiness matrix.

Next runnable E0 command:

```bash
ssh A5000 'cd ~/tung.nx/cs-asr-reft && scripts/a5000/run.sh e0'
```

There is no valid encoder-output E1 command until its verified implementation and exact experiment inputs
are recovered or separately implemented/reviewed. Do not substitute the decoder smoke for that experiment.
