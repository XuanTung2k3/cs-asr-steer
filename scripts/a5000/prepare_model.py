"""Create a verified model view without modifying the pre-existing shared cache.

Small processor/config files are downloaded at the exact revision. Verified existing
weights are symlinked. Refuse to replace any differing file in the managed model view.
"""
from pathlib import Path
import hashlib

from huggingface_hub import HfApi, hf_hub_download
import yaml

ROOT = Path(__file__).resolve().parents[2]
C = yaml.safe_load((ROOT/'configs/a5000/runtime.yaml').read_text())
model = C['model']
source, dest = [Path(model[k]).expanduser() for k in ['source_cache', 'root']]
rev = model['revision']
info = HfApi().model_info('openai/whisper-large-v3', revision=rev, files_metadata=True)
assert info.sha == rev
refs = {f.rfilename: f for f in info.siblings}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


assert sha(source/'model.safetensors') == model['sha256'] == refs['model.safetensors'].lfs.sha256
dest.mkdir(parents=True, exist_ok=True)
for name in ['config.json', 'preprocessor_config.json', 'generation_config.json', 'tokenizer_config.json',
             'tokenizer.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'normalizer.json']:
    p = dest/name
    if not p.exists():
        hf_hub_download('openai/whisper-large-v3', name, revision=rev, local_dir=dest, force_download=True)
    data = p.read_bytes()
    assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == refs[name].blob_id, p
weights = dest/'model.safetensors'
if not weights.exists():
    weights.symlink_to(source/'model.safetensors')
assert sha(weights) == model['sha256']
print(dest)
