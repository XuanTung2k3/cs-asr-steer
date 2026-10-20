"""Resume the exact authoritative short_wav release; excludes dev/test transcripts.

Uses the caller's existing Hugging Face access. Does not request access, accept terms,
copy credentials, or substitute another dataset. Run without HF_HUB_OFFLINE.
"""
from pathlib import Path
import hashlib
import json

from huggingface_hub import HfApi, snapshot_download
import yaml

ROOT = Path(__file__).resolve().parents[2]
C = yaml.safe_load((ROOT/'configs/a5000/runtime.yaml').read_text())
rev = C['data']['revision']
dest = Path(C['data']['root']).expanduser()
patterns = ['README.md', 'data/short_wav/*', 'data/index/short_wav/train/*',
            'data/index/short_wav/dev/wav.scp', 'data/index/short_wav/test/wav.scp',
            'data/index/long_wav/*.txt', 'data/index/total_information/*']
info = HfApi().dataset_info('BAAI/CS-Dialogue', revision=rev, files_metadata=True)
assert info.sha == rev
snapshot_download('BAAI/CS-Dialogue', repo_type='dataset', revision=rev,
                  local_dir=dest, allow_patterns=patterns, max_workers=4)
hashes = {}
for f in info.siblings:
    if not f.rfilename.startswith('data/short_wav/') or not f.lfs:
        continue
    p = dest/f.rfilename
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    assert p.stat().st_size == f.lfs.size
    assert h.hexdigest() == f.lfs.sha256, p
    hashes[f.rfilename] = {'bytes': f.lfs.size, 'sha256': h.hexdigest()}
out = Path(C['output_root']).expanduser()/'data-download.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'hub_id': 'BAAI/CS-Dialogue', 'revision': rev,
                          'license': 'CC-BY-NC-SA-4.0', 'authoritative_lfs_verified': hashes}, indent=2)+'\n')
print(f'{len(hashes)} archive parts verified: {out}')
