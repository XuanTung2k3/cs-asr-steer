#!/usr/bin/env bash
# Reuses tested CUDA dependencies without modifying the shared ASR environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
asr_python="${ASR_PYTHON:-$HOME/tung.nx/asr/bin/python}"
reft_env="${REFT_ENV:-$HOME/tung.nx/reft-a5000}"
"$asr_python" -c 'import torch; assert torch.__version__ == "2.7.1+cu126"; assert torch.cuda.is_available()'
if [[ ! -x "$reft_env/bin/python" ]]; then
  "$asr_python" -m venv "$reft_env"
fi
asr_site=$("$asr_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$reft_env/bin/python" - "$asr_site" <<'PY'
from pathlib import Path
import sys
import sysconfig
p = Path(sysconfig.get_paths()['purelib'])/'asr-base.pth'
content = sys.argv[1]+'\n'
if p.exists() and p.read_text() != content:
    raise SystemExit('Existing inherited dependency path differs; inspect before changing it')
p.write_text(content)
PY
"$reft_env/bin/python" -m pip install transformers==4.57.6 tokenizers==0.22.2 pytest==8.3.5 uroman==1.3.1.1
"$reft_env/bin/python" -m pip install --no-deps -e .
"$reft_env/bin/python" -m pip check
"$reft_env/bin/python" scripts/a5000/capture_environment.py
