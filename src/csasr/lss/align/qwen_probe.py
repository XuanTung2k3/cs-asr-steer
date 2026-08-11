"""Isolated probe for the third aligner. Always terminal, never a false result.

The Qwen3 forced aligner is downloaded but its runtime has not yet produced an
artifact inside a pipeline run. A third independent aligner strengthens the
consensus, but it must not be able to take a stage down: an *infrastructure*
failure here is recorded as `blocked` and the consensus proceeds with two
families.

Three distinctions this file is careful about.

* **Timeout is not a scientific result.** A checkpoint load that hangs consumes
  the whole allocation, so the probe is bounded. Exceeding the bound yields
  `blocked` with `timed_out: true` -- never `completed_no_go`, which would claim
  the aligner was measured and found wanting.
* **A bug is not infrastructure.** Errors that mean "this environment cannot run
  the aligner" become `blocked`; anything else is `failed`, so a defect in our
  own code surfaces as a defect instead of hiding behind a missing dependency.
* **Partial output is quarantined.** Any candidate table from a run that did not
  complete is moved aside rather than left where the cache would reuse it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ...utils.logging import get_logger

log = get_logger(__name__)

#: exceptions that mean "this environment cannot run the aligner" rather than
#: "our code is wrong". Everything else is a defect and is reported as one.
INFRASTRUCTURE_ERRORS = (
    TimeoutError, ImportError, ModuleNotFoundError, FileNotFoundError,
    OSError, MemoryError, NotImplementedError,
)


@dataclass
class QwenProbeResult:
    state: str                      # "ok" | "completed_no_go" | "blocked" | "failed"
    reason: str | None = None
    model_dir: str = ""
    package_version: str | None = None
    candidate_rows: int = 0
    valid_rows: int = 0
    elapsed_seconds: float = 0.0
    deadline_seconds: float = 0.0
    timed_out: bool = False
    quarantined: list[str] = field(default_factory=list)
    published: list[str] = field(default_factory=list)
    process_group: dict[str, Any] = field(default_factory=dict)
    language_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Deadline:
    """Hard wall-clock bound on the probe.

    Loading a 0.6B checkpoint can hang on a cold cache or a bad shard. In a
    chained job that would consume the whole allocation before the stages that
    matter ever run, so the probe is bounded and a timeout is just another
    `blocked`.

    The probe runs in-process -- the adapter is a Python object, not a
    subprocess -- so there is no process group to signal. SIGALRM raising into
    the interpreter is the available mechanism, and it is why `probe_qwen`
    unloads the adapter and empties the CUDA cache in a `finally` that also runs
    when the alarm fires mid-load: that, not a kill, is what stops this probe
    leaking GPU memory into the rest of the job. If the aligner is ever moved
    behind a subprocess, this class is where `killpg` belongs.
    """

    def __init__(self, seconds: float):
        self.seconds = max(0, int(seconds))
        self.armed = False
        self._previous = None

    def __enter__(self):
        if not self.seconds:
            return self
        import signal

        def _raise(_signum, _frame):
            raise TimeoutError(f"qwen probe exceeded its {self.seconds}s deadline")

        try:
            self._previous = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
            self.armed = True
        except (ValueError, AttributeError):        # not the main thread
            self._previous = None
            log.warning("qwen probe deadline could not be armed (not the main "
                        "thread); the probe is unbounded in this process")
        return self

    def __exit__(self, *exc):
        import signal

        try:
            signal.alarm(0)
            if self._previous is not None:
                signal.signal(signal.SIGALRM, self._previous)
        except (ValueError, AttributeError):
            pass
        return False


def _release_gpu(adapter: Any) -> None:
    """Unload the adapter and free its GPU memory, whatever went wrong."""
    if adapter is not None:
        try:
            adapter.unload()
        except Exception:                                  # pragma: no cover
            log.warning("qwen adapter unload failed; freeing the cache anyway")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                      # pragma: no cover
        pass


def _sweep_stale_cache(out_dir: str | Path | None,
                       *, family: str = "qwen_forced_aligner") -> list[str]:
    """Quarantine a cached table when an in-process probe did not complete.

    Only the in-process path needs this. The production path (`probe_qwen_subprocess`)
    writes into an attempt-private directory, so a failed attempt has nothing in
    the cache to quarantine -- which is why it is scoped to this attempt and
    cannot move an older, valid table.
    """
    if out_dir is None:
        return []
    directory = Path(out_dir)
    if not directory.is_dir():
        return []
    moved: list[str] = []
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for path in sorted(directory.glob(f"candidates_{family}.parquet")):
        target = path.with_name(f"{path.stem}.quarantined.{stamp}{path.suffix}")
        try:
            path.rename(target)
            moved.append(str(target))
            log.warning("quarantined a partial probe artifact: %s -> %s", path, target)
        except OSError:                                    # pragma: no cover
            log.warning("could not quarantine %s", path)
    return moved


def attempt_dir(out_dir: str | Path | None) -> Path | None:
    """A private directory for one probe attempt.

    The probe writes here and nowhere else. Nothing it produces is visible to
    the cache until `publish_attempt` moves it, so a timed-out attempt leaves
    the cache exactly as it found it -- there is no partial artifact to
    quarantine because there was never one in a place that mattered.
    """
    if out_dir is None:
        return None
    directory = Path(out_dir) / f".qwen_attempt_{os.getpid()}_{int(time.time())}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def discard_attempt(attempt: Path | None) -> list[str]:
    """Delete an attempt directory. Returns what it contained, for the record."""
    if attempt is None or not attempt.is_dir():
        return []
    contents = [str(p) for p in sorted(attempt.rglob("*")) if p.is_file()]
    shutil.rmtree(attempt, ignore_errors=True)
    if contents:
        log.warning("discarded %d file(s) from an incomplete probe attempt: %s",
                    len(contents), contents)
    return contents


def publish_attempt(attempt: Path | None, out_dir: str | Path | None,
                    *, family: str = "qwen_forced_aligner") -> list[str]:
    """Move a *successful* attempt's outputs into the cache, then clean up."""
    if attempt is None or out_dir is None or not attempt.is_dir():
        return []
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    # only the candidate table is published; the request payload and the input
    # manifest also live in the attempt directory and are not outputs
    for path in sorted(attempt.glob(f"candidates_{family}*.parquet")):
        target = target_dir / path.name
        os.replace(path, target)                 # atomic within one filesystem
        published.append(str(target))
    shutil.rmtree(attempt, ignore_errors=True)
    return published


def _terminate_group(process: "subprocess.Popen", *, grace_seconds: float = 10.0
                     ) -> dict[str, Any]:
    """Kill the probe's whole process group, escalating to SIGKILL.

    The probe is started with `start_new_session=True`, so it and anything it
    spawns share a process group id equal to its pid. Signalling the group is
    what reaches dataloader workers and CUDA helper processes; signalling the
    child alone leaves them holding the GPU.
    """
    import signal
    import subprocess as sp

    outcome = {"terminated": False, "killed": False, "pgid": None}
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError):           # already gone
        return outcome
    outcome["pgid"] = int(pgid)
    for signum, key in ((signal.SIGTERM, "terminated"), (signal.SIGKILL, "killed")):
        try:
            os.killpg(pgid, signum)
            outcome[key] = True
        except (ProcessLookupError, PermissionError):
            break
        try:
            process.wait(timeout=grace_seconds if signum == signal.SIGTERM else 5.0)
            break
        except sp.TimeoutExpired:
            continue
    return outcome


def probe_qwen(cfg: dict, manifest: pd.DataFrame, geometry, identity, *,
               deadline_seconds: float | None = None,
               out_dir: str | Path | None = None) -> QwenProbeResult:
    """Try to run the Qwen aligner on a handful of utterances.

    Terminal in every path. ``out_dir``, when given, is swept for partial
    candidate tables whenever the probe does not complete cleanly.
    """
    qcfg = ((cfg.get("alignment") or {}).get("qwen") or {})
    model_dir = str(qcfg.get("local_model_dir", ""))
    started = time.monotonic()
    if deadline_seconds is None:
        deadline_seconds = float(qcfg.get("probe_deadline_minutes", 30)) * 60.0

    if not model_dir or not Path(model_dir).exists():
        return QwenProbeResult(state="blocked", reason="checkpoint_missing",
                               model_dir=model_dir,
                               deadline_seconds=float(deadline_seconds),
                               quarantined=_sweep_stale_cache(out_dir))

    adapter = None
    try:
        with _Deadline(deadline_seconds):
            from ...nat5h.aligners import Qwen3ForcedAlignerAdapter

            adapter_cfg = dict(cfg)
            adapter_cfg["qwen_aligner"] = {
                **(cfg.get("qwen_aligner") or {}),
                "model_id": qcfg.get("model_id"),
                "local_model_dir": model_dir,
                "dtype": qcfg.get("dtype", "bfloat16"),
                "device": (cfg.get("model") or {}).get("device", "cuda"),
            }
            adapter = Qwen3ForcedAlignerAdapter(adapter_cfg)
            adapter.load()
            n = int(qcfg.get("probe_utterances", 8))
            sample = manifest.head(n)
            table = adapter.run(sample, geometry=geometry, identity=identity)
            rows = int(len(table)) if table is not None else 0
            valid = int(table["is_valid"].sum()) if rows and "is_valid" in table else 0
            elapsed = time.monotonic() - started
            if rows and out_dir is not None:
                # A probe that discards its output proves the aligner works and
                # then leaves Gate A with nothing to use, which is why the
                # second-aligner repair path was not executable.
                target = Path(out_dir) / "candidates_qwen_forced_aligner.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                table.to_parquet(target, index=False)
                log.info("probe wrote %d Qwen candidates to %s", rows, target)
            if rows == 0:
                # the adapter ran to completion and produced nothing: that is a
                # measurement, and the only path to `completed_no_go` here
                return QwenProbeResult(
                    state="completed_no_go",
                    reason="adapter_returned_no_candidates",
                    model_dir=model_dir, elapsed_seconds=elapsed,
                    deadline_seconds=float(deadline_seconds),
                    quarantined=_sweep_stale_cache(out_dir))
            return QwenProbeResult(state="ok", model_dir=model_dir,
                                   candidate_rows=rows, valid_rows=valid,
                                   elapsed_seconds=elapsed,
                                   deadline_seconds=float(deadline_seconds))
    except BaseException as exc:
        elapsed = time.monotonic() - started
        timed_out = isinstance(exc, TimeoutError)
        # infrastructure failures are `blocked`; anything else is a defect in
        # our own code and must not be laundered into a missing-aligner story
        infrastructure = isinstance(exc, INFRASTRUCTURE_ERRORS)
        state = "blocked" if infrastructure else "failed"
        log.warning("Qwen probe %s after %.1f s: %s: %s",
                    "timed out" if timed_out else state, elapsed,
                    type(exc).__name__, exc)
        result = QwenProbeResult(
            state=state,
            reason=("deadline_exceeded" if timed_out
                    else f"{type(exc).__name__}: {exc}"),
            model_dir=model_dir,
            elapsed_seconds=elapsed,
            deadline_seconds=float(deadline_seconds),
            timed_out=timed_out,
            quarantined=_sweep_stale_cache(out_dir),
            traceback=traceback.format_exc(limit=20),
        )
        if not isinstance(exc, Exception):      # KeyboardInterrupt, SystemExit
            raise                               # `finally` still frees the GPU
        return result
    finally:
        _release_gpu(adapter)


#: entry point the child process runs. It re-enters this module in a fresh
#: interpreter, so a hang inside native CUDA code is a hung *process* that can
#: be killed, not a hung thread that SIGALRM may never interrupt.
_CHILD_MAIN = "csasr.lss.align.qwen_probe"


def probe_qwen_subprocess(cfg: dict, manifest: pd.DataFrame, *,
                          deadline_seconds: float | None = None,
                          out_dir: str | Path | None = None,
                          python: str | None = None) -> QwenProbeResult:
    """Run the probe in its own session, bounded by a killable deadline.

    This is the production path. `probe_qwen` remains available for in-process
    use (tests, interactive debugging), but it can only bound Python-level
    hangs; a checkpoint load stuck inside a native call ignores SIGALRM and
    would hold the allocation and the GPU until the job wall-clock expired.

    On timeout the whole process group is signalled -- SIGTERM, then SIGKILL --
    so dataloader workers and CUDA helpers die with it. Nothing the attempt
    wrote is published, because it wrote into an attempt-private directory.
    """
    import json
    import tempfile

    qcfg = ((cfg.get("alignment") or {}).get("qwen") or {})
    model_dir = str(qcfg.get("local_model_dir", ""))
    if deadline_seconds is None:
        deadline_seconds = float(qcfg.get("probe_deadline_minutes", 30)) * 60.0
    started = time.monotonic()

    if not model_dir or not Path(model_dir).exists():
        return QwenProbeResult(state="blocked", reason="checkpoint_missing",
                               model_dir=model_dir,
                               deadline_seconds=float(deadline_seconds))

    attempt = attempt_dir(out_dir) or Path(tempfile.mkdtemp(prefix="qwen_attempt_"))
    payload = attempt / "request.json"
    result_path = attempt / "result.json"
    manifest_path = attempt / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    payload.write_text(json.dumps({
        "cfg": cfg, "manifest": str(manifest_path),
        "result": str(result_path), "out_dir": str(attempt),
    }, default=str), encoding="utf-8")

    process = subprocess.Popen(
        [python or sys.executable, "-m", _CHILD_MAIN, "--request", str(payload)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)          # its own process group, so killpg works

    try:
        output, _ = process.communicate(timeout=float(deadline_seconds))
        code = int(process.returncode)
    except subprocess.TimeoutExpired:
        kill = _terminate_group(process)
        try:
            output, _ = process.communicate(timeout=5.0)
        except Exception:                                   # pragma: no cover
            output = ""
        elapsed = time.monotonic() - started
        discarded = discard_attempt(attempt)
        log.warning("Qwen probe exceeded its %.0f s deadline; process group %s "
                    "terminated=%s killed=%s", deadline_seconds, kill["pgid"],
                    kill["terminated"], kill["killed"])
        return QwenProbeResult(
            state="blocked", reason="deadline_exceeded", model_dir=model_dir,
            elapsed_seconds=elapsed, deadline_seconds=float(deadline_seconds),
            timed_out=True, quarantined=discarded,
            traceback=(output or "")[-2000:] or None,
            process_group=kill)

    elapsed = time.monotonic() - started
    if code != 0 or not result_path.is_file():
        discarded = discard_attempt(attempt)
        # a child that died is infrastructure; a child that ran and reported a
        # defect is a defect, and it says so in its own result payload
        return QwenProbeResult(
            state="blocked", reason=f"probe_process_exit_{code}",
            model_dir=model_dir, elapsed_seconds=elapsed,
            deadline_seconds=float(deadline_seconds), quarantined=discarded,
            traceback=(output or "")[-2000:] or None)

    result = QwenProbeResult(**{
        k: v for k, v in json.loads(result_path.read_text(encoding="utf-8")).items()
        if k in QwenProbeResult.__dataclass_fields__
    })
    result.elapsed_seconds = elapsed
    result.deadline_seconds = float(deadline_seconds)
    if result.state == "ok":
        result.published = publish_attempt(attempt, out_dir)
    else:
        result.quarantined = discard_attempt(attempt)
    return result


def _child_main(argv: list[str] | None = None) -> int:
    """Run one probe attempt and write its result. Never used by the parent."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Qwen probe (child process)")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    from ...nat5h.coordinates import EncoderGeometry
    from ...nat5h.schema import RunIdentity

    cfg = request["cfg"]
    manifest = pd.read_parquet(request["manifest"])
    result = probe_qwen(cfg, manifest, EncoderGeometry.from_values(),
                        RunIdentity.from_cfg(cfg),
                        deadline_seconds=0,          # the parent owns the clock
                        out_dir=request["out_dir"])
    Path(request["result"]).write_text(json.dumps(result.to_dict(), default=str),
                                       encoding="utf-8")
    return 0


def families_after_probe(configured: list[str], result: QwenProbeResult) -> list[str]:
    """Drop the Qwen family unless the probe actually produced candidates.

    `blocked`, `completed_no_go` and `failed` all drop it: a family that did not
    demonstrably work cannot count towards the two-aligner requirement.
    """
    if result.state == "ok":
        return list(configured)
    return [f for f in configured if f != "qwen_forced_aligner"]


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(_child_main())
