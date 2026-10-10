# DIR-SPRINT0 D5 provider — installation and reproduction guide

Rebuilds the frozen provider exactly. No sudo. Nothing is installed into acl1, no system package is touched, and no
historical model cache is used. The hashes of record are in `results/inference_cf/dir_sprint0/d5_provider/provider_manifest.json`.

## 1. Paths and pins

```bash
REV=2c733782da5604684829819a5eb744c193fe9398
MODEL=/mnt/data/tungnx/cs-asr-steer/providers/wav2vec2-xlsr-53-espeak-cv-ft/$REV
ENV=/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5
WHEELS=/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5_wheelhouse
ACL1=/home/tungnx/miniconda3/envs/acl1/bin/python
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH
```

## 2. Model files (pinned revision, direct URLs; no HF cache)

```bash
mkdir -p $MODEL && cd $MODEL
for f in .gitattributes README.md config.json preprocessor_config.json special_tokens_map.json tokenizer_config.json vocab.json pytorch_model.bin; do
  curl -sSL --fail -o "$f" "https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/resolve/$REV/$f"
done
sha256sum pytorch_model.bin   # must be 04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551
```

Compare every file against `provider_manifest.json` → `model.files`. The adapter refuses to load on any mismatch.

**If the server cannot reach huggingface.co**, run the same `curl` loop on a machine with access, then transfer the
files and verify the hashes on arrival. Never copy from an unverified mirror.

```bash
scp -r ./<REV>/ tungnx@<server>:/mnt/data/tungnx/cs-asr-steer/providers/wav2vec2-xlsr-53-espeak-cv-ft/
```

## 3. Isolated environment (acl1 stays read-only)

```bash
$ACL1 -m pip freeze | sha256sum        # record; must be unchanged at the end
$ACL1 -m venv --system-site-packages $ENV
mkdir -p $WHEELS
$ENV/bin/python -m pip download --no-deps -d $WHEELS "panphon==0.22.2" "unicodecsv==0.14.1"
cat > $ENV/requirements-d5.txt <<'EOF'
panphon==0.22.2 --hash=sha256:a4c65113430d0699054cb00df978c02712d3c80913a1ef67697f888d96f3a00a
unicodecsv==0.14.1 --hash=sha256:018c08037d48649a0412063ff4eda26eaa81eff1546dbffa51fa5293276ff7fc
EOF
$ENV/bin/python -m pip install --no-deps --no-index --find-links $WHEELS --require-hashes -r $ENV/requirements-d5.txt
$ACL1 -m pip freeze | sha256sum        # identical to the first value
```

For an offline server, run `pip download` elsewhere and `scp` the two files into `$WHEELS`. `--require-hashes` then
verifies them.

The venv borrows acl1's torch 2.10.0+cu128, transformers 4.57.6, numpy 1.26.4 and soundfile 0.13.1. `pip check` in the
venv reports opencv/rerun/torch conflicts that already exist in acl1; they are unrelated to this provider. The
provider's `Wav2Vec2PhonemeCTCTokenizer` is deliberately unused, and `phonemizer`/espeak must **not** be installed.

## 4. Engineering audio (public; plausibility only)

The URLs, licenses and hashes are in `provider_manifest.json` → `engineering_audio`:
- two LibriSpeech model-card samples;
- two AISHELL-1 examples and one Common Voice zh-CN example from Apache-2.0 SpeechBrain repositories, at pinned
  revisions.

Save them to `/mnt/data/tungnx/cs-asr-steer/providers/engineering_audio/`.

## 5. Rebuild and verify

```bash
cd /home/tungnx/cs-asr-steer-inf
$ENV/bin/python -I experiments/inference_cf_dir_sprint0_d5_feature_table.py      # table_digest must match the manifest
$ENV/bin/python -I experiments/inference_cf_dir_sprint0_d5_provider.py engineering --threads 4
$ENV/bin/python -I experiments/inference_cf_dir_sprint0_d5_provider_audit.py     # DIR_SPRINT0_D5_PROVIDER_AUDIT: PASS
$ENV/bin/python -m pytest -q tests/test_dir_sprint0_d5_provider.py
```

Notes on these steps:
- `provider.py manifest` regenerates the manifest. Run it only for a deliberate, documented provider change, because
  it changes the manifest digest.
- The engineering check's posterior hashes are CPU float32 with 4 threads. A GPU or a different thread count needs
  its own determinism check before use.

## 6. Use

```python
from csasr.inference_cf.phone_provider import PhoneProvider, feature_evidence, interval_weights
prov = PhoneProvider("results/inference_cf/dir_sprint0/d5_provider/provider_manifest.json",
                     "results/inference_cf/dir_sprint0/d5_provider/feature_table.json", device="cpu")
r = prov.posteriors(waveform_16k_mono_float32, 16000)          # full original audio
w = interval_weights(window_start_sample, window_end_sample, r["n_frames"])
ev = feature_evidence(r["log_probs"], w, prov.table)          # ev["q"], ev["mass"], ev["status"]
```
