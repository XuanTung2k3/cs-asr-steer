#!/usr/bin/env bash
# BASIS-A6 batch environment bootstrap.
# Source this file from every A6 sbatch wrapper; do not rely on login-shell state.

set -euo pipefail

export BASIS_A6_ENV_PREFIX="${BASIS_A6_ENV_PREFIX:-/home/tungnx/miniconda3/envs/acl1}"
export BASIS_A6_PYTHON="${BASIS_A6_PYTHON:-${BASIS_A6_ENV_PREFIX}/bin/python}"

if [[ ! -x "${BASIS_A6_PYTHON}" ]]; then
    echo "BASIS-A6 environment missing Python: ${BASIS_A6_PYTHON}" >&2
    return 1 2>/dev/null || exit 1
fi

# The environment-local C++ runtime is required by SciPy/sklearn and must win over
# the batch host's older system libstdc++.  This is process-local, not a system change.
export PATH="${BASIS_A6_ENV_PREFIX}/bin:${PATH}"
if [[ -d "${BASIS_A6_ENV_PREFIX}/lib" ]]; then
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${BASIS_A6_ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${BASIS_A6_ENV_PREFIX}/lib"
    fi
fi

a6_print_environment() {
    local require_cuda="${1:-0}"
    "${BASIS_A6_PYTHON}" - "${require_cuda}" <<'PY'
import importlib
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone

require_cuda = sys.argv[1] == "1"
print("hostname", socket.gethostname())
print("date", datetime.now(timezone.utc).isoformat())
print("which_python", subprocess.check_output(["which", "python"], text=True).strip())
print("sys.executable", sys.executable)
print("python_version", sys.version.replace("\n", " "))
print("git_head", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

try:
    import torch
except Exception as exc:
    raise SystemExit(f"BASIS-A6 environment error: torch import failed: {exc!r}")

print("torch_version", torch.__version__)
print("torch_cuda_version", torch.version.cuda)
print("torch_cuda_available", torch.cuda.is_available())
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("BASIS-A6 environment error: CUDA unavailable for GPU job")
if require_cuda:
    print("visible_gpu", torch.cuda.get_device_name(0))
else:
    print("visible_gpu", "CPU_MODE")

for name in ("scipy", "transformers", "datasets", "sklearn"):
    try:
        module = importlib.import_module(name)
        print(f"{name}_version", getattr(module, "__version__", "<unknown>"))
    except Exception as exc:
        raise SystemExit(f"BASIS-A6 environment error: {name} import failed: {exc!r}")
PY
}

if [[ "${BASIS_A6_ENV_SELFTEST:-0}" == "1" ]]; then
    a6_print_environment "${BASIS_A6_REQUIRE_CUDA:-0}"
fi
