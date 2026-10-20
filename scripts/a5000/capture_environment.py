"""Save the complete resolved environment and system metadata outside Git."""
from pathlib import Path
import importlib.metadata as md
import json
import platform
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
C = yaml.safe_load((ROOT/'configs/a5000/runtime.yaml').read_text())
out = Path(C['output_root']).expanduser()/'environment'
out.mkdir(parents=True, exist_ok=True)


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}


(out/'pip-freeze.txt').write_text(run([sys.executable, '-m', 'pip', 'freeze', '--all'])['stdout'])
base = Path.home()/'tung.nx/asr/bin/python'
(out/'asr-base-pip-freeze.txt').write_text(run([str(base), '-m', 'pip', 'freeze', '--all'])['stdout'])
packages = ['torch', 'torchaudio', 'transformers', 'tokenizers', 'huggingface-hub', 'accelerate',
            'datasets', 'numpy', 'scipy', 'pandas', 'soundfile', 'pyarrow', 'scikit-learn', 'pytest', 'safetensors', 'uroman']
rec = {'python': platform.python_version(), 'executable': sys.executable,
       'base_packages': str(base.parent.parent/'lib/python3.13/site-packages'),
       'environment_type': 'Dedicated venv with read-only ASR dependency path; overrides are environment-local',
       'packages': {p: md.version(p) for p in packages},
       'package_locations': {p: str(md.distribution(p).locate_file('')) for p in packages},
       'torch_build': torch.__version__, 'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
       'platform': platform.platform(), 'pip_check': run([sys.executable, '-m', 'pip', 'check']),
       'system': {name: run(args) for name, args in {
           'gpu': ['nvidia-smi'], 'os': ['cat', '/etc/os-release'], 'cpu': ['lscpu'],
           'ram': ['free', '-h'], 'disk': ['df', '-h', str(Path.home()), str(Path(C['data']['root']).expanduser())],
           'processes': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'],
           'kernel': ['uname', '-a']}.items()}}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    rec['device'] = {'name': p.name, 'bytes': p.total_memory, 'capability': torch.cuda.get_device_capability()}
(out/'metadata.json').write_text(json.dumps(rec, indent=2)+'\n')
print(out)
