# A5000 environment

Recorded 2026-10-20. SSH alias `A5000` resolves to user `ubuntu`, home `/home/ubuntu`,
container hostname `50c00b4c6d3a`. These are resolved remote paths, independent of the H100 layout.

## Host

| Item | Observed value |
|---|---|
| OS / architecture | Ubuntu 20.04.6 LTS / x86_64 |
| Kernel | Linux 5.15.0-190-generic |
| CPU | 2 × Intel Xeon Gold 6242, 64 logical CPUs |
| RAM | 251 GiB total, about 240 GiB initially available |
| GPUs | Six NVIDIA RTX A5000; one GPU used, visible device 0 |
| VRAM | `nvidia-smi`: 24,564 MiB per GPU; PyTorch usable total 25,425,608,704 bytes |
| Compute capability | 8.6; actual FP16/BF16/FP32 CUDA matrix operations passed |
| NVIDIA driver | 535.309.01 |
| Driver-reported CUDA | 12.2 |
| Tested PyTorch runtime | CUDA 12.6, inherited from existing ASR installation |
| Initial GPU usage | 1 MiB per device, no compute processes |
| Scheduling | No `sbatch`, `squeue`, `srun`, `qsub` or `bsub` exposed; standalone container execution |
| Storage | Home overlay initially 633 GiB free; shared filesystem initially 582 GiB, about 565 GiB after setup |

CUDA version reporting above deliberately distinguishes driver capability from the wheel's bundled runtime.
The existing CUDA 12.6 PyTorch build was retained only after actual model and compute checks; no H100-specific
CUDA build or driver upgrade was installed. Bounded outcomes do not establish compatibility of every optional
CUDA kernel or full training campaign.

## Python dependencies and isolation

Existing ASR environment: `/home/ubuntu/tung.nx/asr`, Python **3.13.3**. Its base interpreter lives under
`/home/ubuntu/.local/share/uv/python/cpython-3.13.3-linux-x86_64-gnu/`.
The original environment remains unchanged, including Transformers 4.52.4 and Tokenizers 0.21.1.

Active ReFT environment: `/home/ubuntu/tung.nx/reft-a5000`. This dedicated venv reads existing ASR packages
through its local `asr-base.pth`; overrides and the editable project installation live in the ReFT venv.
This is dependency reuse, **not a self-contained isolated copy**. Reproduction requires the recorded ASR
base or an equivalent reconstruction from its lock. Changing/deleting the base environment can affect ReFT.
No global package installation, sudo, unrelated package upgrade or base-environment mutation was performed.

| Package | Resolved version |
|---|---|
| Python | 3.13.3 |
| pip | 25.0.1 |
| PyTorch / torchaudio | 2.7.1+cu126 / 2.7.1+cu126 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| Hugging Face Hub | 0.35.0 |
| Accelerate / Datasets | 1.7.0 / 3.6.0 |
| NumPy / SciPy / pandas | 2.2.0 / 1.15.3 / 2.2.3 |
| soundfile / pyarrow | 0.13.1 / 20.0.0 |
| scikit-learn / safetensors | 1.6.1 / 0.5.3 |
| pytest / uroman | 8.3.5 / 1.3.1.1 |

`pip check` reports no broken requirements. Transformers 4.57.6 resolves the loader's `dtype` API mismatch.
`uroman` was missing from pyproject's dependencies but is required by an applicable alignment diagnostic test;
installing it fixed that failure. No source formula, mask, objective or score was changed.

Full environment records are outside Git:

```text
/home/ubuntu/shared/tung.nx/a5000-validation/environment/metadata.json
/home/ubuntu/shared/tung.nx/a5000-validation/environment/pip-freeze.txt
/home/ubuntu/shared/tung.nx/a5000-validation/environment/asr-base-pip-freeze.txt
```

The metadata records package locations, Python/torch/CUDA/cuDNN versions, GPU properties, driver, CPU, RAM,
OS, disk and pip consistency. `scripts/a5000/setup_environment.sh` recreates the environment-local overrides;
`capture_environment.py` saves complete resolved versions. `reft-a5000-tools` was a temporary diagnostic venv
using the older base Transformers; it is retained, unused by the final wrapper, to preserve diagnostic logs.

## Model and data roots

Authoritative model view:

```text
/home/ubuntu/shared/tung.nx/models/whisper-large-v3/06f233fe06e710322aca913c1bc4249a0d71fce1
```

Whisper-large-v3 revision: `06f233fe06e710322aca913c1bc4249a0d71fce1`.
Weight SHA-256: `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`.
All ten small processor/config files plus the weight file are checked against authoritative metadata.
The weight is a symlink into the pre-existing verified cache. That cache had changed `pad_token` and
`extra_special_tokens` fields in `tokenizer_config.json`, and changed `pad_token` in `special_tokens_map.json`.
The original cache was preserved; authoritative small files were fetched into the separate pinned view.
`prepare_model.py` and `verify_model.py` reproduce and verify this layout without rewriting the original cache.

Dataset root: `/home/ubuntu/shared/tung.nx/CS-Dialogue`.
Dataset revision: `a84770fcdf14072252373a0227321227eaac7dc7` of `BAAI/CS-Dialogue`.
The published `short_wav` release is 19 concatenated gzip parts, about 9.0 GiB compressed. All parts match
authoritative LFS SHA-256 and size. Training audio is extracted beneath `data/short_wav/extracted/short_wav/`;
about 7.5 GiB is extracted, and the complete local dataset directory occupies about 17 GiB.
The exact index paths match the repository reader. License: CC BY-NC-SA 4.0; API reported the release ungated
and permitted download. No access request or license agreement was submitted by this setup.
See the [authoritative dataset card](https://huggingface.co/datasets/BAAI/CS-Dialogue) for its terms.

Only training audio was extracted and opened by audio readers. The archive stream traversed the release,
skipping non-training members. Dev/test `wav.scp` and long-form split indexes were inventoried by ID;
dev/test transcripts were not downloaded, and dev/test audio was not decoded or scored.

## Execution

All A5000 outputs: `/home/ubuntu/shared/tung.nx/a5000-validation`, outside Git. The wrapper defaults to one GPU,
two CPU threads, offline model loading, and a task-specific temporary directory. Override `CUDA_VISIBLE_DEVICES`
or `REFT_PYTHON` explicitly if needed. No Slurm script or frozen H100 path/config was edited.

```bash
ssh A5000 'cd ~/tung.nx/cs-asr-reft && scripts/a5000/run.sh e0'
ssh A5000 'cd ~/tung.nx/cs-asr-reft && timeout 600 scripts/a5000/run.sh smoke'
```

For downloads/remote metadata checks, invoke `download_data.py`, `prepare_model.py` or `verify_model.py` directly
with `~/tung.nx/reft-a5000/bin/python`, without the wrapper's `HF_HUB_OFFLINE=1`.

Remote rsync was absent. Ubuntu's rsync 3.1.3 and libpopt0 packages were downloaded and extracted without
installation under `/home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted`. The deployment script sets
that executable's library path explicitly. It uses `--update`, preserves Git metadata/executable bits, and
never uses `--delete`; divergent HEADs or tracked remote edits require inspection.
