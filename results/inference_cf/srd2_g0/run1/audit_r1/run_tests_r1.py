"""Run the committed SRD2-G0 focused test suite with the r1 auditor substituted for the canonical auditor module."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
spec = importlib.util.spec_from_file_location("experiments.inference_cf_srd2_g0_audit",
                                              Path(__file__).with_name("inference_cf_srd2_g0_audit_r1.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
import experiments  # noqa: E402
experiments.inference_cf_srd2_g0_audit = mod
assert mod.ROOT == ROOT
sys.exit(pytest.main(["-q", "-p", "no:warnings", "-p", "no:cacheprovider", str(ROOT / "tests/test_inference_cf_srd2_g0.py")]))
