"""Provenance: file hashes, git state, environment, Slurm, weight hashes and write-once protocol freezes.

Freeze records live in <artifact_root>/freeze/. A GPU or evaluation command calls ``check_freeze(stage)`` and refuses to
run if any frozen source changed since the latest freeze of that stage; an amendment is a new ``<stage>_rK.json`` with a
timestamped note naming the changed sources.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import platform
import subprocess
from pathlib import Path

from reft.common import ROOT, art, read_json, write_json

FREEZE_SOURCES = {
    "p0": ["reft/__init__.py", "reft/common.py", "reft/provenance.py", "reft/text.py", "reft/data_audit.py",
           "reft/baseline_auto.py", "configs/reft/p0p1.yaml", "docs/reft/P0P1_PROTOCOL.md", "docs/reft/SCOPE.md",
           "slurm/reft_p0.sbatch", "slurm/reft_cpu.sbatch", "src/csasr/models/whisper.py",
           "src/csasr/data/normalize.py", "src/csasr/data/language_tags.py"],
}
FREEZE_SOURCES["p1"] = FREEZE_SOURCES["p0"] + [
    "reft/site.py", "reft/gradient_capture.py", "reft/build_basis.py", "reft/evaluate_basis.py", "slurm/reft_p1.sbatch"]


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_paths": [l[3:] for l in run("status", "--porcelain").splitlines()]}


def slurm() -> dict:
    return {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME",
                                           "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK")}


def environment() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform(), "hostname": platform.node()}
    try:
        import torch
        import transformers
        out.update(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out.update(gpu=p.name, gpu_total_memory_bytes=int(p.total_memory))
    except ImportError:
        pass
    return out


def loaded_cuda_libs() -> dict:
    """Paths of the CUDA libraries mapped into this process (call after the first GPU matmul)."""
    import re
    out = {}
    try:
        for line in open("/proc/self/maps"):
            m = re.search(r"(/\S*/(lib(cublasLt|cublas|cudart|cudnn|cufft|nvJitLink)\.so[.\d]*))$", line.strip())
            if m:
                out.setdefault(m.group(3), set()).add(m.group(1))
    except OSError:
        pass
    return {k: sorted(v) for k, v in out.items()}


def cublas_consistent(libs: dict) -> bool:
    """libcublas and libcublasLt (and libcudart, when mapped) resolve from one pip package tree, one version each."""
    import os
    a, lt = libs.get("cublas", []), libs.get("cublasLt", [])
    if len(a) != 1 or len(lt) != 1 or os.path.dirname(a[0]) != os.path.dirname(lt[0]):
        return False
    return all(len(v) == 1 for v in libs.values())


def weights_hash(model) -> str:
    """Bit-exact hash of every parameter in its bfloat16 form (float32 copies of bf16 weights cast back exactly)."""
    import torch
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode())
        t = p.detach().to("cpu")
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)
        h.update(t.view(torch.int16 if t.dtype == torch.bfloat16 else t.dtype).numpy().tobytes())
    return "sha256:" + h.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ freeze

def _freeze_dir() -> Path:
    return art("freeze")


def latest_freeze(stage: str) -> Path:
    d = _freeze_dir()
    am = sorted(d.glob(f"{stage}_r*.json"), key=lambda p: int(p.stem.split("_r")[1]))
    return am[-1] if am else d / f"{stage}.json"


def freeze(stage: str, note: str | None = None) -> Path:
    rec = {"schema": "reft_freeze_v1", "stage": stage, "utc": now(), "git": git_state(),
           "sources": {p: sha(ROOT / p) for p in FREEZE_SOURCES[stage]}}
    base = _freeze_dir() / f"{stage}.json"
    if not base.exists():
        return write_json(base, rec)
    if not note:
        raise SystemExit("an amendment freeze needs --note (timestamped justification)")
    prev = latest_freeze(stage)
    k = 1 if prev == base else int(prev.stem.split("_r")[1]) + 1
    prev_src = read_json(prev)["sources"]
    rec.update(supersedes=prev.name, supersedes_sha256=sha(prev), amendment_note=note,
               changed_sources=sorted(p for p, h in rec["sources"].items() if prev_src.get(p) != h))
    return write_json(_freeze_dir() / f"{stage}_r{k}.json", rec)


def check_freeze(stage: str) -> dict:
    fz = latest_freeze(stage)
    if not fz.exists():
        raise SystemExit(f"no freeze for stage {stage}")
    rec = read_json(fz)
    bad = [p for p, h in rec["sources"].items() if sha(ROOT / p) != h]
    if bad:
        raise SystemExit(f"sources changed since {fz.name}: {bad}")
    return {"freeze": fz.name, "freeze_sha256": sha(fz), "freeze_git_head": rec["git"]["head"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "check"])
    ap.add_argument("--stage", required=True, choices=sorted(FREEZE_SOURCES))
    ap.add_argument("--note")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        print(freeze(a.stage, a.note))
    else:
        print(check_freeze(a.stage))


if __name__ == "__main__":
    main()
