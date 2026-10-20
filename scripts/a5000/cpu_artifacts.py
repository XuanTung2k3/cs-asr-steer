"""Run the existing artifact-dependent tokenizer tests against the portable model path.

Only this process's configuration lookup changes; frozen files remain untouched.
"""
import copy
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from reft import common

C = yaml.safe_load((ROOT/'configs/a5000/runtime.yaml').read_text())
original_cfg = common.cfg


def portable_cfg():
    c = copy.deepcopy(original_cfg())
    c['model']['path'] = str(Path(C['model']['root']).expanduser())
    return c


common.cfg = portable_cfg
out = Path(C['output_root']).expanduser()/'cpu'
out.mkdir(parents=True, exist_ok=True)
raise SystemExit(pytest.main([
    str(ROOT/'tests/reft/test_d1_regionkl.py')+'::test_real_whisper_tokenizer_regions_are_han_only_and_match_windows',
    str(ROOT/'tests/reft/test_d2_edit_locality.py')+'::test_real_whisper_tokenizer_mandarin_tokens_are_han_only',
    '--junitxml='+str(out/'tokenizer.xml'),
]))
