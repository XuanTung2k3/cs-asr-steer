"""Compare local model/processor files with authoritative metadata at the pinned revision."""
from pathlib import Path
import hashlib
import json

from huggingface_hub import HfApi
import yaml

ROOT = Path(__file__).resolve().parents[2]
C = yaml.safe_load((ROOT/'configs/a5000/runtime.yaml').read_text())
root = Path(C['model']['root']).expanduser()
info = HfApi().model_info('openai/whisper-large-v3', revision=C['model']['revision'], files_metadata=True)
assert info.sha == C['model']['revision']
files = {f.rfilename: f for f in info.siblings}
names = ['model.safetensors', 'config.json', 'preprocessor_config.json', 'generation_config.json',
         'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json', 'added_tokens.json',
         'vocab.json', 'merges.txt', 'normalizer.json']
verified = {}
for name in names:
    p, ref = root/name, files[name]
    size = p.stat().st_size
    h = hashlib.sha256()
    git = hashlib.sha1(f'blob {size}\0'.encode())
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
            git.update(b)
    if ref.lfs:
        assert h.hexdigest() == ref.lfs.sha256 and size == ref.lfs.size, name
    else:
        assert git.hexdigest() == ref.blob_id, name
    verified[name] = {'sha256': h.hexdigest(), 'bytes': size, 'authoritative_match': True}
out = Path(C['output_root']).expanduser()/'model-verification.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'model': 'openai/whisper-large-v3', 'revision': info.sha,
                          'root': str(root), 'files': verified}, indent=2)+'\n')
print(f'{len(verified)} model/processor files match revision {info.sha}: {out}')
