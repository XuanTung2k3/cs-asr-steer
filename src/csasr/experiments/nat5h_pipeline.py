"""Resumable NAT5H exploratory natural-audio feasibility pipeline."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from csasr.directions.accumulators import DirectionAccumulatorSet
from csasr.directions.controls import random_direction
from csasr.directions.encoder import build_direction_record, load_direction, save_direction
from csasr.evaluation.correction_harm import correction_harm_table, summarize_correction_harm
from csasr.evaluation.mer import corpus_mer
from csasr.evaluation.pier import evaluate_pois
from csasr.models.generation import decode_manifest
from csasr.models.hook_tests import run_hook_tests
from csasr.models.hooks import ActivationRecorder, assert_no_hooks
from csasr.models.whisper import batch_features, load_audio, load_whisper
from csasr.nat5h.aligners import (
    Qwen3ForcedAlignerAdapter,
    reference_units_table,
    run_existing_ctc,
    run_whisper_dtw,
)
from csasr.nat5h.consensus import (
    ConsensusCriteria,
    N1ConsensusGate,
    SmokeCriteria,
    build_consensus,
    build_unit_consensus_v2,
    evaluate_n1_consensus_gate,
    smoke_passes,
    summarize_alignment_smoke,
    summarize_alignment_smoke_v2,
)
from csasr.nat5h.coordinates import (
    EncoderGeometry,
    sample_span_to_encoder_span,
    waveform_num_frames,
)
from csasr.nat5h.model_cache import ensure_qwen_checkpoint
from csasr.nat5h.schema import (
    ALIGNMENT_SCHEMA_VERSION,
    RunIdentity,
    artifact_matches_identity,
    assert_schema_v2,
)
from csasr.nat5h.selection import (
    reject_dev_confirm,
    reject_test_split,
    select_pois,
    speaker_balanced_sample,
    validate_disjoint_speakers,
)
from csasr.nat5h.statusing import (
    StageTimer,
    atomic_output_path,
    atomic_write_json,
    atomic_write_text,
    read_status,
    stage_status_payload,
    terminal_state,
    write_stage_status,
)
from csasr.nat5h.units import build_reference_units, transcript_has_bilingual_content
from csasr.steering.encoder_hook import EncoderLocalSteering, global_encoder_steering
from csasr.steering.masks import MaskSpec, ms_to_frames
from csasr.utils.config import load_config
from csasr.utils.hashing import manifest_hash, sha256_file, sha256_obj
from csasr.utils.provenance import source_snapshot_hash

STAGES = [
    "n0_preflight",
    "n1_align_smoke",
    "n2_consensus",
    "n3_direction_mini",
    "n4_availability_mini",
    "n5_steering_select",
    "n6_steering_check",
    "n7_summary",
]

STOP_REQUESTED = False


def _handle_stop(signum, frame) -> None:  # pragma: no cover - exercised by Slurm
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGUSR1, _handle_stop)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    with atomic_output_path(path, suffix=".parquet.tmp") as tmp:
        df.to_parquet(tmp, index=False)
    return path


def atomic_write_json_file(payload: dict[str, Any], path: str | Path) -> Path:
    atomic_write_json(path, payload)
    return Path(path)


def report(path: str | Path, title: str, body: str) -> Path:
    text = f"# {title}\n\n{body.rstrip()}\n"
    atomic_write_text(path, text)
    return Path(path)


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "no_git_repository"


class Nat5HPipeline:
    def __init__(self, cfg: dict, *, mode: str = "run") -> None:
        self.cfg = cfg
        self.mode = mode
        self.root = Path(cfg["experiment"]["output_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("status", "reports", "metrics", "alignments", "directions", "manifests", "predictions", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        (self.root / "diagnostics").mkdir(parents=True, exist_ok=True)
        self.config_hash = sha256_obj(cfg)
        self.code_commit = code_commit()
        self.identity = RunIdentity.from_cfg(cfg, self.code_commit)
        self.source_hash = source_snapshot_hash()
        self.started = time.monotonic()
        self.max_wall = float(cfg["experiment"].get("max_wall_minutes", 300)) * 60.0
        self.reserve = float(cfg["experiment"].get("summary_reserve_minutes", 30)) * 60.0
        self._bundle = None

    # ------------------------------------------------------------------
    # shared data/model helpers
    # ------------------------------------------------------------------
    def remaining_seconds(self) -> float:
        return self.max_wall - (time.monotonic() - self.started)

    def should_stop_before_large_stage(self) -> bool:
        return STOP_REQUESTED or self.remaining_seconds() < self.reserve

    def load_manifest(self, split: str) -> pd.DataFrame:
        path = Path(self.cfg["data"][f"{split}_manifest"])
        df = pd.read_parquet(path)
        reject_test_split(df)
        if not self.cfg["data"].get("use_dev_confirm", False):
            reject_dev_confirm(df)
        return df

    def train_manifest(self) -> pd.DataFrame:
        return self.load_manifest("train")

    def dev_manifest(self) -> pd.DataFrame:
        return self.load_manifest("dev_select")

    def baseline_dev_select(self) -> pd.DataFrame:
        return pd.read_parquet(self.cfg["data"]["baseline_dev_select"])

    def bundle(self):
        if self._bundle is None:
            self._bundle = load_whisper(self.cfg)
        return self._bundle

    def release_bundle(self) -> None:
        self._bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def geometry(self) -> EncoderGeometry:
        return EncoderGeometry.from_bundle(self.bundle())

    def manifest_hashes(self) -> dict[str, str]:
        out = {}
        for key in ("train_manifest", "dev_select_manifest", "baseline_dev_select", "primary_baseline_metadata"):
            p = Path(self.cfg["data"][key])
            if p.exists():
                out[key] = sha256_file(p)
        return out

    def write_status(
        self,
        *,
        stage: str,
        state: str,
        timer: StageTimer,
        output_paths: list[str] | None = None,
        gate_evidence: dict[str, Any] | None = None,
        next_permitted_stage: str | None = None,
        reason: str | None = None,
    ) -> None:
        payload = stage_status_payload(
            stage=stage,
            state=state,
            timer=timer,
            config_hash=self.config_hash,
            code_commit=self.code_commit,
            input_manifest_hashes=self.manifest_hashes(),
            output_paths=output_paths or [],
            gate_evidence=gate_evidence or {},
            next_permitted_stage=next_permitted_stage,
            failure_or_no_go_reason=reason,
            exploratory=True,
        )
        write_stage_status(self.root, payload)

    def skip_if_terminal(self, stage: str, overwrite: bool) -> bool:
        state = terminal_state(self.root, stage)
        return bool(state and not overwrite)

    # ------------------------------------------------------------------
    # stage N0
    # ------------------------------------------------------------------
    def n0_preflight(self) -> str:
        stage = "n0_preflight"
        timer = StageTimer()
        out_report = self.root / "reports" / "n0_preflight.md"
        evidence: dict[str, Any] = {
            "exploratory": True,
            "alignment_schema_version": ALIGNMENT_SCHEMA_VERSION,
            "production_artifacts_root": self.cfg["experiment"]["production_artifacts_root"],
            "new_output_root": str(self.root),
            "source_snapshot_hash": self.source_hash,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
        }
        try:
            train = self.train_manifest()
            dev = self.dev_manifest()
            reject_test_split(train)
            reject_test_split(dev)
            validate_disjoint_speakers(train, dev)
            evidence.update(
                train_rows=int(len(train)),
                dev_select_rows=int(len(dev)),
                train_speakers=int(train["speaker_id"].nunique()),
                dev_select_speakers=int(dev["speaker_id"].nunique()),
            )
            for pkey in ("train_manifest", "dev_select_manifest", "baseline_dev_select", "primary_baseline_metadata"):
                p = Path(self.cfg["data"][pkey])
                evidence[f"{pkey}_exists"] = p.exists()
                if not p.exists():
                    raise FileNotFoundError(f"missing required input: {p}")

            import huggingface_hub  # noqa: F401
            import qwen_asr  # noqa: F401
            import transformers  # noqa: F401
            import importlib.metadata as im

            evidence["python"] = subprocess.check_output(["python", "--version"], text=True).strip()
            evidence["torch_version"] = torch.__version__
            evidence["transformers_version"] = transformers.__version__
            evidence["huggingface_hub_version"] = huggingface_hub.__version__
            evidence["qwen_asr_version"] = im.version("qwen-asr")
            evidence["cuda_available"] = bool(torch.cuda.is_available())
            evidence["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            if self.cfg["model"]["device"] == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")

            try:
                import torchaudio  # noqa: F401

                evidence["torchaudio_import"] = "ok"
            except Exception as exc:
                evidence["torchaudio_import"] = "failed"
                evidence["torchaudio_error"] = repr(exc)

            model_path = Path(self.cfg["model"]["id"])
            evidence["whisper_checkpoint_exists"] = model_path.exists()
            if not model_path.exists():
                raise FileNotFoundError(f"Whisper checkpoint missing: {model_path}")

            usage = shutil.disk_usage(str(self.root.parent))
            evidence["disk_free_gb"] = round(usage.free / (1024**3), 2)
            if usage.free < 100 * (1024**3):
                raise RuntimeError(f"low disk space under {self.root.parent}: {usage.free}")

            # Slurm-side model acquisition.  If compute nodes lack network and
            # Qwen is absent, this stage records a blocked outcome and stops.
            qres = ensure_qwen_checkpoint(
                model_id=self.cfg["qwen_aligner"]["model_id"],
                local_model_dir=self.cfg["qwen_aligner"]["local_model_dir"],
                cache_dir=self.cfg["model_cache"]["hf_hub_cache"],
                marker_path=self.root / "models" / "qwen3_forced_aligner_complete.json",
                allow_download=True,
                lock_timeout_seconds=int(self.cfg["qwen_aligner"].get("lock_timeout_seconds", 7200)),
            )
            evidence["qwen_checkpoint"] = qres["state"]
            state = "passed"
            reason = None
        except Exception as exc:
            state = "blocked" if "Qwen checkpoint" in str(exc) or "download" in str(exc).lower() else "technical_failed"
            reason = repr(exc)
            evidence["error"] = reason

        run_identity_path = atomic_write_json_file(
            {
                "alignment_schema_version": ALIGNMENT_SCHEMA_VERSION,
                "code_commit": self.code_commit,
                "config_hash": self.config_hash,
                "source_snapshot_hash": self.source_hash,
                "resolved_config": self.cfg,
                "evidence": evidence,
            },
            self.root / "diagnostics" / "run_identity.json",
        )

        body = "\n".join(
            [
                "N0 validates environment, split isolation, cache paths, and model availability.",
                "",
                f"- state: `{state}`",
                f"- reason: `{reason}`" if reason else "- reason: none",
                f"- torch: `{evidence.get('torch_version')}`",
                f"- cuda: `{evidence.get('cuda_device')}`",
                f"- torchaudio import: `{evidence.get('torchaudio_import')}`",
                f"- output root: `{self.root}`",
            ]
        )
        report(out_report, "NAT5H N0 Preflight", body)
        self.write_status(
            stage=stage,
            state=state,
            timer=timer,
            output_paths=[str(out_report), str(run_identity_path)],
            gate_evidence=evidence,
            next_permitted_stage="n1_align_smoke" if state == "passed" else "n7_summary",
            reason=reason,
        )
        return state

    # ------------------------------------------------------------------
    # N1 / N2 alignment
    # ------------------------------------------------------------------
    def _bilingual_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        max_dur = self.cfg.get("data", {}).get("max_candidate_duration_sec")
        if max_dur is not None and "duration_sec" in df.columns:
            df = df[df["duration_sec"].astype(float) <= float(max_dur)].copy()
        keep = []
        for _, row in df.iterrows():
            units = build_reference_units(str(row["utterance_id"]), row.get("transcript_raw", ""))
            if transcript_has_bilingual_content(units):
                keep.append(True)
            else:
                keep.append(False)
        return df[keep].reset_index(drop=True)

    def select_smoke_manifest(self) -> pd.DataFrame:
        dev = self._bilingual_candidates(self.dev_manifest())
        max_dur = self.cfg.get("alignment", {}).get("n1_max_duration_sec")
        if max_dur is not None and len(dev):
            short = dev[dev["duration_sec"].astype(float) <= float(max_dur)].copy()
            if len(short) >= 20:
                dev = short
        if dev.empty:
            return dev
        tmp = []
        for _, row in dev.iterrows():
            units = build_reference_units(str(row["utterance_id"]), row.get("transcript_raw", ""))
            en = sum(1 for u in units if u.language == "EN")
            zh = sum(1 for u in units if u.language == "ZH")
            other = sum(1 for u in units if u.language == "OTHER")
            tmp.append({**row.to_dict(), "_en": en, "_zh": zh, "_other": other, "_variety": abs(en - 3)})
        cand = pd.DataFrame(tmp)
        cand = cand[cand["_other"] <= 2]
        if cand.empty:
            cand = pd.DataFrame(tmp)
        cand = cand.sort_values(["_variety", "duration_sec", "utterance_id"])
        return speaker_balanced_sample(cand.drop(columns=["_en", "_zh", "_other", "_variety"], errors="ignore"), 20, seed=42)

    def run_aligners(self, manifest: pd.DataFrame, active_families: set[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        # Derive geometry from Whisper, then release it before CTC/Qwen.
        geometry = self.geometry()
        frames: list[pd.DataFrame] = []
        diagnostics: dict[str, Any] = {"alignment_schema_version": ALIGNMENT_SCHEMA_VERSION}
        requested = set(self.cfg["alignment"].get("candidate_aligners", []))
        if "whisper_dtw" in requested and (active_families is None or "whisper_dtw" in active_families):
            dtw_rows, dtw_diag = run_whisper_dtw(self.bundle(), manifest, self.cfg, self.identity)
            frames.append(dtw_rows)
            diagnostics["whisper_dtw"] = dtw_diag
            atomic_write_json(
                self.root / "diagnostics" / "whisper_dtw_mapping_summary.json",
                dtw_diag,
            )
            self.release_bundle()
        if "existing_ctc" in requested and (active_families is None or "existing_ctc" in active_families):
            try:
                frames.append(run_existing_ctc(manifest, self.cfg, geometry, self.identity))
            except Exception as exc:
                atomic_write_json(
                    self.root / "diagnostics" / "existing_ctc_unavailable.json",
                    {"state": "blocked", "error": repr(exc)},
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if "qwen3_forced_aligner" in requested and (active_families is None or "qwen_forced_aligner" in active_families):
            if not bool(self.cfg.get("qwen_aligner", {}).get("run_during_align_only", False)):
                atomic_write_json(
                    self.root / "diagnostics" / "qwen_adapter_smoke.json",
                    {
                        "state": "blocked",
                        "reason": "qwen_runtime_deferred_to_qwen_smoke_mode",
                        "model_id": self.cfg["qwen_aligner"]["model_id"],
                        "local_model_dir": self.cfg["qwen_aligner"]["local_model_dir"],
                        "detail": (
                            "Qwen checkpoint exists, but full Qwen loading/first diagnostic did not "
                            "produce an artifact within the alignment-debug polling window. "
                            "Run `sbatch cs_asr_nat5h.sh qwen_smoke` for isolated Qwen diagnostics."
                        ),
                    },
                )
                diagnostics["qwen_forced_aligner"] = {"state": "blocked", "reason": "qwen_runtime_deferred_to_qwen_smoke_mode"}
                out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                if len(out):
                    assert_schema_v2(out, artifact="run_aligners output")
                return out, diagnostics
            qwen = Qwen3ForcedAlignerAdapter(self.cfg)
            languages = qwen.supported_languages()
            if active_families is not None:
                chosen = self._selected_qwen_variant()
                languages = [chosen] if chosen else []
            qwen_diag = {
                "installed_package_version": qwen.package_version,
                "model_id": qwen.model_id,
                "local_model_dir": qwen.local_model_dir,
                "language_diagnostics": [],
            }
            for lang in languages:
                try:
                    if len(manifest):
                        qwen_diag["language_diagnostics"].append(qwen.diagnostic_smoke(manifest.head(1), language=lang))
                        atomic_write_json(self.root / "diagnostics" / "qwen_adapter_smoke.json", qwen_diag)
                    frames.append(
                        qwen.run(
                            manifest,
                            geometry,
                            language=lang,
                            identity=self.identity,
                            diagnostics_dir=self.root / "diagnostics",
                        )
                    )
                except Exception as exc:
                    qwen_diag["language_diagnostics"].append({"language": lang, "error": repr(exc)})
                    atomic_write_json(
                        self.root / "diagnostics" / f"qwen3_{lang}_unavailable.json",
                        {"state": "blocked", "language": lang, "error": repr(exc)},
                    )
            atomic_write_json(self.root / "diagnostics" / "qwen_adapter_smoke.json", qwen_diag)
            qwen.unload()
        frames = [f for f in frames if f is not None and len(f)]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(out):
            assert_schema_v2(out, artifact="run_aligners output")
        return out, diagnostics

    def n1_align_smoke(self) -> str:
        stage = "n1_align_smoke"
        timer = StageTimer()
        manifest = self.select_smoke_manifest()
        manifest_path = atomic_write_parquet(manifest, self.root / "manifests" / "n1_smoke_manifest.parquet")
        refs = reference_units_table(manifest)
        refs_path = atomic_write_parquet(refs, self.root / "alignments" / "smoke" / "reference_units.parquet")
        candidates, diag = self.run_aligners(manifest)
        assert_schema_v2(candidates, artifact="N1 canonical candidates")
        cand_path = atomic_write_parquet(candidates, self.root / "metrics" / "n1_canonical_candidates.parquet")
        # compatibility copy for human inspection
        align_path = atomic_write_parquet(candidates, self.root / "alignments" / "smoke" / "all_aligners_v2.parquet")

        metrics = summarize_alignment_smoke_v2(candidates, refs)
        metrics_path = atomic_write_parquet(metrics, self.root / "metrics" / "n1_aligner_summary.parquet")
        # Retain legacy filename as diagnostic table, not as the gate.
        atomic_write_parquet(metrics, self.root / "metrics" / "n1_aligner_smoke.parquet")

        criteria = ConsensusCriteria(
            min_aligners=int(self.cfg["alignment"]["consensus_min_aligners"]),
            max_start_disagreement_ms=float(self.cfg["alignment"]["max_start_disagreement_ms"]),
            max_end_disagreement_ms=float(self.cfg["alignment"]["max_end_disagreement_ms"]),
            min_unit_coverage=float(self.cfg["alignment"]["min_unit_coverage"]),
            # N1 is a mechanical compatibility smoke.  The unchanged 200 ms
            # boundary-agreement test is the gate here; 250 ms safe-interior
            # erosion is an N2 downstream adequacy question.
            safe_interior_erosion_ms=0.0,
            min_safe_interior_ms=1.0,
            steering_union_padding_ms=float(self.cfg["alignment"]["steering_union_padding_ms"]),
        )
        consensus, rejected, pairwise = build_unit_consensus_v2(candidates, criteria, reference_units_df=refs)
        consensus_path = atomic_write_parquet(consensus, self.root / "metrics" / "n1_consensus_spans.parquet")
        rejected_path = atomic_write_parquet(rejected, self.root / "metrics" / "n1_rejected_spans.parquet")
        pairwise_path = atomic_write_parquet(pairwise, self.root / "metrics" / "n1_pairwise_disagreements.parquet")

        gate_cfg = self.cfg["alignment"].get("n1_gate", {})
        gate = N1ConsensusGate(
            min_consensus_utterances=int(gate_cfg.get("min_consensus_utterances", 10)),
            min_consensus_spans=int(gate_cfg.get("min_consensus_spans", 50)),
            min_consensus_en_spans=int(gate_cfg.get("min_consensus_en_spans", 20)),
            min_consensus_zh_spans=int(gate_cfg.get("min_consensus_zh_spans", 20)),
            min_valid_families=int(gate_cfg.get("min_valid_families", 2)),
        )
        state, gate_evidence = evaluate_n1_consensus_gate(consensus, candidates, gate)
        reason = ";".join(gate_evidence["failure_reasons"]) if state != "passed" else None
        next_stage = "n2_consensus" if state == "passed" else "n7_summary"
        active_families = sorted(set(sum((x for x in consensus.get("selected_aligner_families", pd.Series(dtype=object)).tolist()), []))) if len(consensus) else []
        if not active_families:
            active_families = sorted(candidates.loc[candidates["is_valid"], "aligner_family"].dropna().unique().tolist())
        atomic_write_json_file({"active_families": active_families}, self.root / "metrics" / "n1_active_aligner_families.json")

        # Select one Qwen language variant mechanically if Qwen produced valid rows.
        qwen_summary = {}
        qwen_metrics = metrics[metrics["aligner_family"] == "qwen_forced_aligner"] if len(metrics) else pd.DataFrame()
        if len(qwen_metrics):
            qwen_metrics = qwen_metrics.sort_values(["valid_units", "unit_coverage", "aligner_variant"], ascending=[False, False, True])
            selected_variant = str(qwen_metrics.iloc[0]["aligner_variant"])
            qwen_summary["selected_language"] = selected_variant.split("/", 1)[1] if "/" in selected_variant else None
        qdiag_path = self.root / "diagnostics" / "qwen_adapter_smoke.json"
        if qdiag_path.exists() and qwen_summary:
            payload = json.loads(qdiag_path.read_text(encoding="utf-8"))
            payload.update(qwen_summary)
            atomic_write_json(qdiag_path, payload)

        failure_counts = {
            "candidate_failure_counts": candidates.loc[~candidates["is_valid"], "failure_code"].value_counts().to_dict() if len(candidates) else {},
            "consensus_rejection_counts": rejected["rejection_code"].value_counts().to_dict() if len(rejected) else {},
            "n1_gate": gate_evidence,
        }
        failure_counts_path = atomic_write_json_file(failure_counts, self.root / "diagnostics" / "alignment_failure_counts.json")

        body = [
            "N1 checks mechanical compatibility on 20 automatically selected dev_select bilingual utterances.",
            "",
            "Gate is now reference-unit-level consensus, not whole-aligner global pass/fail.",
            "",
            f"- selected utterances: {len(manifest)}",
            f"- valid candidate families: {', '.join(gate_evidence['valid_families']) if gate_evidence['valid_families'] else 'none'}",
            f"- accepted consensus spans: {gate_evidence['consensus_spans']}",
            f"- accepted consensus utterances: {gate_evidence['consensus_utterances']}",
            f"- accepted EN/ZH spans: {gate_evidence['consensus_en_spans']} / {gate_evidence['consensus_zh_spans']}",
            f"- state: `{state}`",
            f"- reason: `{reason}`" if reason else "- reason: none",
        ]
        if len(metrics):
            body.append("")
            body.append(metrics.to_markdown(index=False))
        out_report = report(self.root / "reports" / "n1_align_smoke.md", "NAT5H N1 Alignment Smoke", "\n".join(body))
        self.write_status(
            stage=stage,
            state=state,
            timer=timer,
            output_paths=[str(manifest_path), str(refs_path), str(cand_path), str(align_path), str(metrics_path), str(consensus_path), str(rejected_path), str(pairwise_path), str(failure_counts_path), str(out_report)],
            gate_evidence=gate_evidence,
            next_permitted_stage=next_stage,
            reason=reason,
        )
        return state

    def _valid_aligners(self) -> set[str]:
        path = self.root / "metrics" / "n1_valid_aligners.json"
        if not path.exists():
            return set()
        return set(json.loads(path.read_text(encoding="utf-8")).get("valid_aligners", []))

    def _active_families(self) -> set[str]:
        path = self.root / "metrics" / "n1_active_aligner_families.json"
        if not path.exists():
            return set()
        return set(json.loads(path.read_text(encoding="utf-8")).get("active_families", []))

    def _selected_qwen_variant(self) -> str | None:
        path = self.root / "diagnostics" / "qwen_adapter_smoke.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload.get("selected_language")
        return str(selected) if selected else None

    def n2_consensus(self) -> str:
        stage = "n2_consensus"
        timer = StageTimer()
        active = self._active_families()
        if len(active) < int(self.cfg["alignment"]["consensus_min_aligners"]):
            self.write_status(stage=stage, state="completed_no_go", timer=timer, reason="n1_did_not_produce_two_active_aligner_families", next_permitted_stage="n7_summary")
            return "completed_no_go"

        train = self._bilingual_candidates(self.train_manifest())
        dev = self._bilingual_candidates(self.dev_manifest())
        train = speaker_balanced_sample(train, int(self.cfg["data"]["train_candidate_utterances"]), seed=42)
        dev = speaker_balanced_sample(dev, int(self.cfg["data"]["dev_candidate_utterances"]), seed=43)
        train_manifest_path = atomic_write_parquet(train, self.root / "manifests" / "n2_train_candidates.parquet")
        dev_manifest_path = atomic_write_parquet(dev, self.root / "manifests" / "n2_dev_select_candidates.parquet")

        train_cand_path = self.root / "alignments" / "n2_candidates_train_v2.parquet"
        dev_cand_path = self.root / "alignments" / "n2_candidates_dev_select_v2.parquet"

        def _load_or_run_candidates(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
            if path.exists():
                cached = pd.read_parquet(path)
                if artifact_matches_identity(cached, self.identity):
                    return cached
            rows, _ = self.run_aligners(frame, active_families=active)
            assert_schema_v2(rows, artifact=str(path))
            atomic_write_parquet(rows, path)
            return rows

        train_align = _load_or_run_candidates(train_cand_path, train)
        dev_align = _load_or_run_candidates(dev_cand_path, dev)

        criteria = ConsensusCriteria(
            min_aligners=int(self.cfg["alignment"]["consensus_min_aligners"]),
            max_start_disagreement_ms=float(self.cfg["alignment"]["max_start_disagreement_ms"]),
            max_end_disagreement_ms=float(self.cfg["alignment"]["max_end_disagreement_ms"]),
            min_unit_coverage=float(self.cfg["alignment"]["min_unit_coverage"]),
            safe_interior_erosion_ms=float(self.cfg["alignment"]["safe_interior_erosion_ms"]),
            min_safe_interior_ms=float(self.cfg["alignment"]["min_safe_interior_ms"]),
            steering_union_padding_ms=float(self.cfg["alignment"]["steering_union_padding_ms"]),
        )
        train_refs = reference_units_table(train)
        dev_refs = reference_units_table(dev)
        consensus_train, rejected_train, agree_train = build_unit_consensus_v2(train_align, criteria, reference_units_df=train_refs)
        consensus_dev, rejected_dev, agree_dev = build_unit_consensus_v2(dev_align, criteria, reference_units_df=dev_refs)
        ctrain_path = atomic_write_parquet(consensus_train, self.root / "alignments" / "consensus_train.parquet")
        cdev_path = atomic_write_parquet(consensus_dev, self.root / "alignments" / "consensus_dev_select.parquet")
        combined_consensus = pd.concat([
            consensus_train.assign(consensus_split="train"),
            consensus_dev.assign(consensus_split="dev_select"),
        ], ignore_index=True)
        combined_rejected = pd.concat([
            rejected_train.assign(consensus_split="train"),
            rejected_dev.assign(consensus_split="dev_select"),
        ], ignore_index=True)
        n2_consensus_path = atomic_write_parquet(combined_consensus, self.root / "metrics" / "n2_consensus_spans.parquet")
        n2_rejected_path = atomic_write_parquet(combined_rejected, self.root / "metrics" / "n2_rejected_spans.parquet")
        agree_path = atomic_write_parquet(pd.concat([agree_train.assign(consensus_split="train"), agree_dev.assign(consensus_split="dev_select")], ignore_index=True), self.root / "metrics" / "n2_aligner_agreement.parquet")

        consensus_dev_for_poi = consensus_dev.copy()
        try:
            wrong_pois, correct_pois = select_pois(
                consensus_dev_for_poi,
                dev,
                self.baseline_dev_select(),
                wrong=9999,
                correct=9999,
                seed=92,
            )
            wrong_poi_count = int(len(wrong_pois))
            correct_poi_count = int(len(correct_pois))
        except Exception:
            wrong_poi_count = 0
            correct_poi_count = 0

        counts = {
            "alignment_schema_version": ALIGNMENT_SCHEMA_VERSION,
            "active_families": sorted(active),
            "train_candidate_utterances": int(len(train)),
            "dev_candidate_utterances": int(len(dev)),
            "accepted_train_spans": int(len(consensus_train)),
            "accepted_dev_spans": int(len(consensus_dev)),
            "accepted_train_bilingual_utterances": int(consensus_train.groupby("utterance_id")["language"].nunique().ge(2).sum()) if len(consensus_train) else 0,
            "dev_safe_en_spans": int((consensus_dev["language"] == "EN").sum()) if len(consensus_dev) else 0,
            "dev_safe_zh_spans": int((consensus_dev["language"] == "ZH").sum()) if len(consensus_dev) else 0,
            "wrong_english_pois": wrong_poi_count,
            "correct_english_pois": correct_poi_count,
            "train_speakers": int(train["speaker_id"].nunique()) if len(train) else 0,
            "dev_speakers": int(dev["speaker_id"].nunique()) if len(dev) else 0,
            "train_conversations": int(train["conversation_id"].nunique()) if "conversation_id" in train else 0,
            "dev_conversations": int(dev["conversation_id"].nunique()) if "conversation_id" in dev else 0,
            "consensus_units_by_language": combined_consensus["language"].value_counts().to_dict() if len(combined_consensus) else {},
            "rejection_counts": combined_rejected["rejection_code"].value_counts().to_dict() if len(combined_rejected) else {},
        }
        enough_representation = (
            counts["accepted_train_bilingual_utterances"] >= 300
            and counts["dev_safe_en_spans"] >= 100
            and counts["dev_safe_zh_spans"] >= 100
        )
        enough_steering = counts["wrong_english_pois"] >= 50 and counts["correct_english_pois"] >= 50
        if enough_representation and enough_steering:
            branch = "A_full_local_pilot_viable"
            state = "passed"
            reason = None
            next_stage = "n3_direction_mini"
        elif enough_representation:
            branch = "B_representation_pilot_only"
            state = "passed"
            reason = "insufficient_pois_for_local_steering"
            next_stage = "n3_direction_mini"
        else:
            branch = "C_consensus_remains_insufficient"
            state = "insufficient_data"
            reason = "natural_consensus_counts_below_minimum_desired"
            next_stage = "n7_summary"
        counts["branch"] = branch
        body = [
            "N2 builds natural pseudo-alignments at reference-unit level when at least two distinct aligner families agree within 200 ms.",
            "",
            f"- state: `{state}`",
            f"- branch: `{branch}`",
            f"- reason: `{reason}`" if reason else "- reason: none",
            "",
            "```json",
            json.dumps(counts, indent=2),
            "```",
        ]
        out_report = report(self.root / "reports" / "n2_consensus_report.md", "NAT5H N2 Natural Consensus", "\n".join(body))
        # compatibility name
        report(self.root / "reports" / "n2_consensus.md", "NAT5H N2 Natural Consensus", "\n".join(body))
        self.write_status(
            stage=stage,
            state=state,
            timer=timer,
            output_paths=[str(train_manifest_path), str(dev_manifest_path), str(train_cand_path), str(dev_cand_path), str(ctrain_path), str(cdev_path), str(n2_consensus_path), str(n2_rejected_path), str(agree_path), str(out_report)],
            gate_evidence=counts,
            next_permitted_stage=next_stage,
            reason=reason,
        )
        # Required r2 filename. Keep the payload identical except the stage name.
        n2_payload = read_status(self.root, stage)
        if n2_payload:
            duplicate = dict(n2_payload)
            duplicate["stage"] = "n2_consensus_status"
            atomic_write_json(self.root / "status" / "n2_consensus_status.json", duplicate)
        atomic_write_json(
            self.root / "status" / "pipeline_state.json",
            {
                "alignment_schema_version": ALIGNMENT_SCHEMA_VERSION,
                "n2_state": state,
                "branch": branch,
                "next_permitted_stage": next_stage,
                "resume_after_alignment_allowed": bool(state == "passed"),
            },
        )
        return state

    # ------------------------------------------------------------------
    # N3 directions
    # ------------------------------------------------------------------
    def _add_frame_columns(self, consensus: pd.DataFrame, geometry: EncoderGeometry) -> pd.DataFrame:
        out = consensus.copy()
        sf, ef, ms, me = [], [], [], []
        for _, r in out.iterrows():
            enc = waveform_num_frames(int(r["waveform_num_samples"]), geometry)
            a, b = sample_span_to_encoder_span(int(r["safe_start_sample"]), int(r["safe_end_sample"]), geometry, enc)
            c, d = sample_span_to_encoder_span(int(r["mask_start_sample"]), int(r["mask_end_sample"]), geometry, enc)
            sf.append(a); ef.append(b); ms.append(c); me.append(d)
        out["safe_start_frame"] = sf
        out["safe_end_frame"] = ef
        out["mask_start_frame"] = ms
        out["mask_end_frame"] = me
        return out

    def _frame_indices_for_utt(self, consensus_utt: pd.DataFrame, language: str, n_valid: int) -> np.ndarray:
        keep = np.zeros(n_valid, dtype=bool)
        sub = consensus_utt[consensus_utt["language"] == language]
        for _, r in sub.iterrows():
            s, e = int(r["safe_start_frame"]), int(r["safe_end_frame"])
            if e > s:
                keep[max(0, s) : min(n_valid, e)] = True
        return np.flatnonzero(keep)

    @torch.inference_mode()
    def _accumulate_nat_directions(self, seed: int, manifest: pd.DataFrame, consensus: pd.DataFrame) -> DirectionAccumulatorSet:
        bundle = self.bundle()
        layers = list(map(int, self.cfg["direction"]["layers"]))
        ckpt = self.root / "directions" / "checkpoints" / f"nat5h_seed{seed}.npz"
        accs = DirectionAccumulatorSet.load_or_new(ckpt, layers, bundle.d_model)
        by_utt = {u: g for u, g in consensus.groupby("utterance_id")}
        work = manifest[manifest["utterance_id"].isin(by_utt) & ~manifest["utterance_id"].isin(accs.processed)].copy()
        work = speaker_balanced_sample(work, len(work), seed=seed)
        bs = int(self.cfg["direction"].get("batch_size", 4))
        checkpoint_every = int(self.cfg["direction"].get("checkpoint_every", 50))
        since = 0
        for start in range(0, len(work), bs):
            if self.should_stop_before_large_stage():
                break
            batch = work.iloc[start : start + bs]
            plans = []
            for _, row in batch.iterrows():
                utt = row["utterance_id"]
                n_valid = bundle.valid_frames(row["duration_sec"])
                g = by_utt.get(utt)
                if g is None:
                    accs.mark(utt)
                    continue
                en = self._frame_indices_for_utt(g, "EN", n_valid)
                zh = self._frame_indices_for_utt(g, "ZH", n_valid)
                if len(en) < int(self.cfg["direction"]["min_en_frames"]) or len(zh) < int(self.cfg["direction"]["min_zh_frames"]):
                    accs.mark(utt)
                    continue
                plans.append((row, en, zh))
            if not plans:
                continue
            features = batch_features(bundle, [p[0]["audio_path"] for p in plans])
            with ActivationRecorder(bundle, layers, module="encoder") as rec:
                bundle.model.model.encoder(features)
                states = {l: rec.states[l] for l in layers}
            assert_no_hooks(bundle)
            for layer in layers:
                h = states[layer]
                acc = accs.acc[layer]
                for i, (row, en_idx, zh_idx) in enumerate(plans):
                    en_t = h[i, torch.as_tensor(en_idx, device=h.device)].double().cpu().numpy()
                    zh_t = h[i, torch.as_tensor(zh_idx, device=h.device)].double().cpu().numpy()
                    both = np.concatenate([en_t, zh_t], axis=0)
                    acc.add_utterance(en_t, zh_t)
                    acc.add_frames(both, gram=both.T @ both)
            del states, features
            for row, _, _ in plans:
                accs.mark(row["utterance_id"])
            since += len(plans)
            if since >= checkpoint_every:
                accs.save(ckpt)
                since = 0
        accs.save(ckpt)
        return accs

    def n3_direction_mini(self) -> str:
        stage = "n3_direction_mini"
        timer = StageTimer()
        train_manifest = pd.read_parquet(self.root / "manifests" / "n2_train_candidates.parquet")
        consensus = pd.read_parquet(self.root / "alignments" / "consensus_train.parquet")
        geometry = self.geometry()
        consensus = self._add_frame_columns(consensus, geometry)
        seeds = list(map(int, self.cfg["direction"]["seeds"]))
        layers = list(map(int, self.cfg["direction"]["layers"]))
        paths = []
        records_by_seed: dict[int, dict[int, dict]] = {}
        stats: dict[str, Any] = {"layers": layers, "seeds": seeds}
        for seed in seeds:
            accs = self._accumulate_nat_directions(seed, train_manifest, consensus)
            records_by_seed[seed] = {}
            for layer, acc in accs.acc.items():
                rec = build_direction_record(
                    self.bundle(),
                    layer,
                    acc.mean_delta(),
                    acc,
                    direction_type="within_utterance_consensus",
                    seed=seed,
                    manifest_hash=manifest_hash(train_manifest),
                    normalization_version=self.cfg["experiment"]["normalization_version"],
                )
                path = save_direction(rec, self.root / "directions")
                paths.append(str(path))
                records_by_seed[seed][layer] = rec
                # diagnostic/global/wrong/random for seed 42 only
                if seed == seeds[0]:
                    grec = build_direction_record(
                        self.bundle(),
                        layer,
                        acc.global_delta(),
                        acc,
                        direction_type="global_centroid_diagnostic",
                        seed=seed,
                        manifest_hash=manifest_hash(train_manifest),
                        normalization_version=self.cfg["experiment"]["normalization_version"],
                    )
                    paths.append(str(save_direction(grec, self.root / "directions")))
                    w = dict(rec)
                    w["direction"] = -rec["direction"]
                    w["direction_type"] = "wrong_sign"
                    paths.append(str(save_direction(w, self.root / "directions")))
                    for rseed in self.cfg["direction"]["random_seeds"]:
                        rr = dict(rec)
                        rr["direction"] = random_direction(rec["direction"].numel(), seed=int(rseed) + int(layer))
                        rr["direction_type"] = f"random_s{int(rseed)}"
                        rr["seed"] = seed
                        rr["control_seed"] = int(rseed)
                        paths.append(str(save_direction(rr, self.root / "directions")))
            stats[f"seed_{seed}_accumulator_summary"] = accs.summary()
        cosines = {}
        if len(seeds) >= 2:
            for layer in layers:
                a = records_by_seed[seeds[0]][layer]["direction"].float()
                b = records_by_seed[seeds[1]][layer]["direction"].float()
                cosines[str(layer)] = float(torch.dot(a, b).item())
        stats["cross_seed_cosine"] = cosines
        stable_layers = [int(l) for l, c in cosines.items() if c >= float(self.cfg["direction"]["min_cross_seed_cosine"])] if cosines else layers
        stats_path = atomic_write_json_file(stats, self.root / "metrics" / "n3_direction_statistics.json")
        if not stable_layers:
            state = "completed_no_go"
            reason = "no_layer_met_cross_seed_cosine_gate"
            next_stage = "n7_summary"
        else:
            state = "passed"
            reason = None
            next_stage = "n4_availability_mini"
        body = [
            "N3 constructs preliminary EN-minus-ZH encoder directions from train speakers only.",
            "",
            f"- state: `{state}`",
            f"- stable layers: {stable_layers}",
            "",
            "```json",
            json.dumps({"cross_seed_cosine": cosines}, indent=2),
            "```",
        ]
        out_report = report(self.root / "reports" / "n3_direction_mini.md", "NAT5H N3 Direction Mini", "\n".join(body))
        self.write_status(stage=stage, state=state, timer=timer, output_paths=paths + [str(stats_path), str(out_report)], gate_evidence=stats, next_permitted_stage=next_stage, reason=reason)
        return state

    # ------------------------------------------------------------------
    # N4 availability
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _projection_rows(self, manifest: pd.DataFrame, consensus: pd.DataFrame, layer: int, direction: torch.Tensor) -> pd.DataFrame:
        bundle = self.bundle()
        by_utt = {u: g for u, g in consensus.groupby("utterance_id")}
        work = manifest[manifest["utterance_id"].isin(by_utt)].copy()
        bs = int(self.cfg["availability"].get("batch_size", 4))
        rows = []
        d = direction.to(bundle.device, torch.float32)
        for start in range(0, len(work), bs):
            if self.should_stop_before_large_stage():
                break
            batch = work.iloc[start : start + bs]
            features = batch_features(bundle, batch["audio_path"].tolist())
            with ActivationRecorder(bundle, [layer], module="encoder") as rec:
                bundle.model.model.encoder(features)
                h = rec.states[layer]
            assert_no_hooks(bundle)
            for i, (_, row) in enumerate(batch.iterrows()):
                utt = row["utterance_id"]
                n_valid = bundle.valid_frames(row["duration_sec"])
                g = by_utt[utt]
                for lang, label in (("EN", 1), ("ZH", 0)):
                    idx = self._frame_indices_for_utt(g, lang, n_valid)
                    if len(idx) == 0:
                        continue
                    proj = (h[i, torch.as_tensor(idx, device=h.device)].float() @ d).detach().cpu().numpy()
                    # cap per utterance/language so one long file cannot dominate
                    if len(proj) > 200:
                        pick = np.linspace(0, len(proj) - 1, 200).astype(int)
                        proj = proj[pick]
                    rows.extend(
                        {
                            "utterance_id": utt,
                            "speaker": row["speaker_id"],
                            "layer": int(layer),
                            "language": lang,
                            "label": label,
                            "projection": float(v),
                        }
                        for v in proj
                    )
        return pd.DataFrame(rows)

    def _availability_metrics(self, proj: pd.DataFrame) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        if proj.empty or proj["label"].nunique() < 2:
            return {"auroc": float("nan"), "auprc": float("nan"), "macro_f1": float("nan"), "speaker_macro_auroc": float("nan")}
        y = proj["label"].astype(int).to_numpy()
        score = proj["projection"].astype(float).to_numpy()
        thr = float(np.median(score))
        pred = (score > thr).astype(int)
        speaker_aurocs = []
        for _, g in proj.groupby("speaker"):
            if g["label"].nunique() == 2:
                speaker_aurocs.append(float(roc_auc_score(g["label"], g["projection"])))
        return {
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "en_recall": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "zh_recall": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "speaker_macro_auroc": float(np.mean(speaker_aurocs)) if speaker_aurocs else float("nan"),
            "num_frames": int(len(proj)),
        }

    def n4_availability_mini(self) -> str:
        stage = "n4_availability_mini"
        timer = StageTimer()
        dev_manifest = pd.read_parquet(self.root / "manifests" / "n2_dev_select_candidates.parquet")
        consensus = pd.read_parquet(self.root / "alignments" / "consensus_dev_select.parquet")
        consensus = self._add_frame_columns(consensus, self.geometry())
        seed = int(self.cfg["direction"]["seeds"][0])
        random_seed = int(self.cfg["direction"]["random_seeds"][0])
        rows = []
        all_proj = []
        # Hook tests on one deterministic utterance before projection/steering.
        sample = dev_manifest.head(1)
        hook_evidence = {}
        if len(sample):
            layer0 = int(self.cfg["direction"]["layers"][0])
            rec0 = load_direction(self.root / "directions", "encoder", layer0, "within_utterance_consensus", seed)
            features = batch_features(self.bundle(), sample["audio_path"].tolist())
            hook_evidence = run_hook_tests(self.bundle(), features, layer0, rec0["direction"].to(self.bundle().device), scale=float(rec0["projection_std"] or 1.0), tol=0.0)
            if not hook_evidence.get("all_passed", False):
                reason = "whisper_hook_invariants_failed"
                atomic_write_json_file(hook_evidence, self.root / "metrics" / "n4_hook_tests.json")
                self.write_status(stage=stage, state="technical_failed", timer=timer, gate_evidence=hook_evidence, next_permitted_stage="n7_summary", reason=reason)
                return "technical_failed"
        atomic_write_json_file(hook_evidence, self.root / "metrics" / "n4_hook_tests.json")

        for layer in map(int, self.cfg["direction"]["layers"]):
            rec = load_direction(self.root / "directions", "encoder", layer, "within_utterance_consensus", seed)
            proj = self._projection_rows(dev_manifest, consensus, layer, rec["direction"])
            if len(proj):
                proj["direction_type"] = "within_utterance_consensus"
                all_proj.append(proj)
            m = self._availability_metrics(proj)
            rand = load_direction(self.root / "directions", "encoder", layer, f"random_s{random_seed}", seed)
            rproj = self._projection_rows(dev_manifest, consensus, layer, rand["direction"])
            rm = self._availability_metrics(rproj)
            rows.append({**m, "layer": layer, "random_auroc": rm.get("auroc"), "margin_above_random": m.get("speaker_macro_auroc", np.nan) - rm.get("speaker_macro_auroc", np.nan)})
        metrics = pd.DataFrame(rows)
        metrics_path = atomic_write_parquet(metrics, self.root / "metrics" / "n4_layer_metrics.parquet")
        if all_proj:
            atomic_write_parquet(pd.concat(all_proj, ignore_index=True), self.root / "metrics" / "n4_projection_rows.parquet")
        metrics = metrics.sort_values(["speaker_macro_auroc", "margin_above_random"], ascending=False)
        selected = metrics.iloc[0].to_dict() if len(metrics) else {}
        selected_layer = int(selected.get("layer", -1)) if selected else -1
        selected_path = atomic_write_json_file({"selected_layer": selected_layer, "selection": selected}, self.root / "reports" / "n4_selected_layer.json")
        gate_pass = bool(selected and selected.get("speaker_macro_auroc", 0.0) >= float(self.cfg["availability"]["min_auroc"]) and selected.get("margin_above_random", -1.0) > 0)
        if gate_pass:
            state = "passed"
            reason = None
            next_stage = "n5_steering_select"
        else:
            state = "completed_no_go"
            reason = "availability_gate_failed"
            next_stage = "n7_summary"
        summary = {"selected_layer": selected_layer, "gate_pass": gate_pass, "selected": selected}
        summary_path = atomic_write_json_file(summary, self.root / "metrics" / "n4_summary.json")
        body = [
            "N4 tests whether the NAT5H EN-minus-ZH direction separates held-out dev_select safe frames.",
            "",
            f"- state: `{state}`",
            f"- selected layer: {selected_layer}",
            f"- speaker-macro AUROC: {selected.get('speaker_macro_auroc') if selected else None}",
            "",
            pd.DataFrame(rows).to_markdown(index=False) if rows else "No projection metrics.",
        ]
        out_report = report(self.root / "reports" / "n4_availability_mini.md", "NAT5H N4 Availability Mini", "\n".join(body))
        self.write_status(stage=stage, state=state, timer=timer, output_paths=[str(metrics_path), str(selected_path), str(summary_path), str(out_report)], gate_evidence=summary, next_permitted_stage=next_stage, reason=reason)
        return state

    # ------------------------------------------------------------------
    # N5/N6 steering
    # ------------------------------------------------------------------
    def _selected_layer_alpha(self) -> tuple[int, float | None]:
        sel = json.loads((self.root / "reports" / "n4_selected_layer.json").read_text(encoding="utf-8"))
        alpha_path = self.root / "metrics" / "n5_selected_config.json"
        alpha = None
        if alpha_path.exists():
            alpha = float(json.loads(alpha_path.read_text(encoding="utf-8"))["alpha"])
        return int(sel["selected_layer"]), alpha

    def _poi_manifests(self, wrong_n: int, correct_n: int, seed: int, exclude=()) -> tuple[pd.DataFrame, pd.DataFrame]:
        consensus = pd.read_parquet(self.root / "alignments" / "consensus_dev_select.parquet")
        consensus = self._add_frame_columns(consensus, self.geometry())
        manifest = pd.read_parquet(self.root / "manifests" / "n2_dev_select_candidates.parquet")
        baseline = self.baseline_dev_select()
        return select_pois(consensus, manifest, baseline, wrong=wrong_n, correct=correct_n, seed=seed, exclude_utterance_ids=exclude)

    def _target_manifest(self, wrong: pd.DataFrame, correct: pd.DataFrame) -> pd.DataFrame:
        manifest = pd.read_parquet(self.root / "manifests" / "n2_dev_select_candidates.parquet")
        ids = pd.concat([wrong, correct], ignore_index=True)["utterance_id"].drop_duplicates()
        return manifest[manifest["utterance_id"].isin(ids)].copy()

    def _spans_from_pois(self, pois: pd.DataFrame) -> dict[str, tuple[int, int]]:
        return {str(r["utterance_id"]): (int(r["mask_start_frame"]), int(r["mask_end_frame"])) for _, r in pois.iterrows()}

    def _random_location_spans(self, pois: pd.DataFrame, manifest: pd.DataFrame, seed: int) -> dict[str, tuple[int, int]]:
        rng = np.random.default_rng(seed)
        spans = {}
        by_utt = manifest.set_index("utterance_id")
        consensus = pd.read_parquet(self.root / "alignments" / "consensus_dev_select.parquet")
        consensus = self._add_frame_columns(consensus, self.geometry())
        consensus_by_utt = {u: g for u, g in consensus.groupby("utterance_id")}
        for _, r in pois.iterrows():
            utt = str(r["utterance_id"])
            n_valid = self.bundle().valid_frames(float(by_utt.loc[utt, "duration_sec"]))
            length = max(1, int(r["mask_end_frame"]) - int(r["mask_start_frame"]))
            forbidden = (int(r["mask_start_frame"]), int(r["mask_end_frame"]))

            # Preferred control: same-duration span inside ZH-labeled consensus
            # speech and disjoint from the POI/pseudo-mask shoulders.
            candidates: list[tuple[int, int]] = []
            for _, zr in consensus_by_utt.get(utt, pd.DataFrame()).iterrows():
                if zr.get("language") != "ZH":
                    continue
                a, b = int(zr["safe_start_frame"]), int(zr["safe_end_frame"])
                for s in range(a, max(a, b - length + 1)):
                    e = s + length
                    if e <= forbidden[0] or s >= forbidden[1]:
                        candidates.append((s, e))

            # Fallback: deterministic non-silent frame spans from short_wav
            # energy, still disjoint from the POI and its shoulders.
            if not candidates:
                try:
                    audio = load_audio(by_utt.loc[utt, "audio_path"], self.bundle().sample_rate)
                    step = int(round(self.bundle().encoder_step_sec * self.bundle().sample_rate))
                    energy = []
                    for f in range(n_valid):
                        chunk = audio[f * step : min(len(audio), (f + 1) * step)]
                        energy.append(float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0)
                    threshold = float(np.percentile([x for x in energy if x > 0] or [0.0], 40))
                    speech = np.asarray(energy) > threshold
                except Exception:
                    speech = np.ones(n_valid, dtype=bool)
                for s in range(0, max(1, n_valid - length + 1)):
                    e = s + length
                    if (e <= forbidden[0] or s >= forbidden[1]) and bool(speech[s:e].all()):
                        candidates.append((s, e))

            if not candidates:
                for s in range(0, max(1, n_valid - length + 1)):
                    e = s + length
                    if e <= forbidden[0] or s >= forbidden[1]:
                        candidates.append((s, e))
            spans[utt] = candidates[int(rng.integers(0, len(candidates)))] if candidates else (0, min(n_valid, length))
        return spans

    def _score_steering_system(self, pois: pd.DataFrame, target_manifest: pd.DataFrame, predictions: pd.DataFrame, system: str) -> tuple[pd.DataFrame, dict]:
        baseline = self.baseline_dev_select().set_index("utterance_id")
        pred = predictions.set_index("utterance_id")
        man = target_manifest.set_index("utterance_id")
        rows = []
        for _, p in pois.iterrows():
            utt = p["utterance_id"]
            base_hyp = baseline.loc[utt, "hypothesis_normalized"]
            steered_hyp = pred.loc[utt, "hypothesis_normalized"]
            rows.append(
                {
                    "utterance_id": utt,
                    "speaker_id": p.get("speaker"),
                    "poi_index": int(p["unit_id"]),
                    "reference": man.loc[utt, "transcript_raw"],
                    "baseline_hypothesis": base_hyp,
                    "steered_hypothesis": steered_hyp,
                    "baseline_correct": bool(p["baseline_correct"]),
                    "baseline_category": p.get("baseline_status", ""),
                    "system": system,
                }
            )
        table = correction_harm_table(rows)
        summary = summarize_correction_harm(table)
        refs = [man.loc[u, "transcript_raw"] for u in pred.index if u in man.index]
        hyps = [pred.loc[u, "hypothesis_normalized"] for u in pred.index if u in man.index]
        summary.update(corpus_mer(refs, hyps))
        # sample PIER on selected POIs
        pier_items = []
        for _, p in pois.iterrows():
            utt = p["utterance_id"]
            pier_items.extend(evaluate_pois(man.loc[utt, "transcript_raw"], pred.loc[utt, "hypothesis_normalized"], [int(p["unit_id"])]))
        summary["sample_pier"] = float(np.mean([not x.correct for x in pier_items])) if pier_items else float("nan")
        summary["system"] = system
        return table, summary

    def _decode_steered(
        self,
        target_manifest: pd.DataFrame,
        pois: pd.DataFrame,
        *,
        layer: int,
        alpha: float,
        direction_type: str,
        system: str,
        spans: dict[str, tuple[int, int]] | None = None,
    ) -> pd.DataFrame:
        seed = int(self.cfg["direction"]["seeds"][0])
        rec = load_direction(self.root / "directions", "encoder", layer, direction_type, seed)
        spec = MaskSpec(kind="EXACT_TAPER", shoulder_frames=ms_to_frames(float(self.cfg["alignment"]["taper_ms"]), self.bundle().encoder_step_sec))
        spans = spans or self._spans_from_pois(pois)
        builder = EncoderLocalSteering(
            self.bundle(),
            layer,
            rec["direction"],
            alpha,
            float(rec.get("projection_std") or 1.0),
            spec,
            spans,
            norm_preserve=bool(self.cfg["steering"]["norm_preserve"]),
        )
        return decode_manifest(
            self.bundle(),
            target_manifest,
            self.cfg,
            system=system,
            config_id=f"{system}_L{layer}_a{alpha}",
            out_path=self.root / "predictions" / f"{system}.parquet",
            language=self.cfg["decoding"].get("language"),
            hook_builder=builder,
            batch_size=int(self.cfg["steering"].get("batch_size", 4)),
            resume=True,
            overwrite=False,
        )

    def n5_steering_select(self) -> str:
        stage = "n5_steering_select"
        timer = StageTimer()
        layer, _ = self._selected_layer_alpha()
        wrong, correct = self._poi_manifests(int(self.cfg["steering"]["select_wrong"]), int(self.cfg["steering"]["select_correct"]), 52)
        wrong_path = atomic_write_parquet(wrong, self.root / "manifests" / "n5_wrong.parquet")
        correct_path = atomic_write_parquet(correct, self.root / "manifests" / "n5_correct.parquet")
        if len(wrong) < int(self.cfg["steering"]["select_wrong"]) or len(correct) < int(self.cfg["steering"]["select_correct"]):
            reason = "insufficient_baseline_wrong_or_correct_pois_for_alpha_selection"
            self.write_status(stage=stage, state="insufficient_data", timer=timer, output_paths=[str(wrong_path), str(correct_path)], gate_evidence={"wrong": len(wrong), "correct": len(correct)}, next_permitted_stage="n7_summary", reason=reason)
            return "insufficient_data"
        target = self._target_manifest(wrong, correct)
        pois = pd.concat([wrong, correct], ignore_index=True)
        results = []
        detail_tables = []
        for alpha in self.cfg["steering"]["alpha_select"]:
            system = f"N5_LOCAL_A{str(alpha).replace('.', 'p')}"
            pred = self._decode_steered(target, pois, layer=layer, alpha=float(alpha), direction_type="within_utterance_consensus", system=system)
            table, summary = self._score_steering_system(pois, target, pred, system)
            summary["alpha"] = float(alpha)
            results.append(summary)
            table["alpha"] = float(alpha)
            detail_tables.append(table)
        res = pd.DataFrame(results).sort_values(["sample_pier", "raw_corrected_count", "raw_corrupted_count", "outside_edit_rate", "alpha"], ascending=[True, False, True, True, True])
        selected = res.iloc[0].to_dict()
        res_path = atomic_write_parquet(res, self.root / "metrics" / "n5_alpha_results.parquet")
        atomic_write_parquet(pd.concat(detail_tables, ignore_index=True), self.root / "metrics" / "n5_per_example.parquet")
        sel_path = atomic_write_json_file({"selected_layer": layer, "alpha": float(selected["alpha"]), "selection": selected}, self.root / "metrics" / "n5_selected_config.json")
        body = [
            "N5 selects alpha on a small dev_select set before held-out steering.",
            "",
            f"- selected layer: {layer}",
            f"- selected alpha: {selected['alpha']}",
            "",
            res.to_markdown(index=False),
        ]
        out_report = report(self.root / "reports" / "n5_steering_select.md", "NAT5H N5 Steering Alpha Selection", "\n".join(body))
        self.write_status(stage=stage, state="passed", timer=timer, output_paths=[str(wrong_path), str(correct_path), str(res_path), str(sel_path), str(out_report)], gate_evidence={"selected_alpha": float(selected["alpha"]), "wrong": len(wrong), "correct": len(correct)}, next_permitted_stage="n6_steering_check")
        return "passed"

    def n6_steering_check(self) -> str:
        stage = "n6_steering_check"
        timer = StageTimer()
        layer, alpha = self._selected_layer_alpha()
        assert alpha is not None
        n5_ids = set(pd.read_parquet(self.root / "manifests" / "n5_wrong.parquet")["utterance_id"].astype(str)) | set(pd.read_parquet(self.root / "manifests" / "n5_correct.parquet")["utterance_id"].astype(str))
        wrong, correct = self._poi_manifests(int(self.cfg["steering"]["check_wrong"]), int(self.cfg["steering"]["check_correct"]), 62, exclude=n5_ids)
        wrong_path = atomic_write_parquet(wrong, self.root / "manifests" / "n6_wrong.parquet")
        correct_path = atomic_write_parquet(correct, self.root / "manifests" / "n6_correct.parquet")
        if len(wrong) < int(self.cfg["steering"]["check_wrong"]) or len(correct) < int(self.cfg["steering"]["check_correct"]):
            state = "insufficient_data"
            reason = "insufficient_disjoint_baseline_wrong_or_correct_pois_for_heldout_check"
        else:
            state = "passed"
            reason = None
        target = self._target_manifest(wrong, correct)
        pois = pd.concat([wrong, correct], ignore_index=True)
        systems = []
        preds = {}
        if len(pois):
            baseline_cache = self.baseline_dev_select()
            preds["N6_CACHED_BASELINE"] = baseline_cache[
                baseline_cache["utterance_id"].isin(target["utterance_id"])
            ].copy()
            preds["N6_LOCAL_CORRECT"] = self._decode_steered(target, pois, layer=layer, alpha=alpha, direction_type="within_utterance_consensus", system="N6_LOCAL_CORRECT")
            preds["N6_LOCAL_WRONG"] = self._decode_steered(target, pois, layer=layer, alpha=alpha, direction_type="wrong_sign", system="N6_LOCAL_WRONG")
            preds["N6_LOCAL_RANDOM"] = self._decode_steered(target, pois, layer=layer, alpha=alpha, direction_type=f"random_s{int(self.cfg['direction']['random_seeds'][0])}", system="N6_LOCAL_RANDOM")
            rloc = self._random_location_spans(pois, target, seed=72)
            preds["N6_RANDOM_LOCATION"] = self._decode_steered(target, pois, layer=layer, alpha=alpha, direction_type="within_utterance_consensus", system="N6_RANDOM_LOCATION", spans=rloc)
            rec = load_direction(self.root / "directions", "encoder", layer, "within_utterance_consensus", int(self.cfg["direction"]["seeds"][0]))
            global_builder = global_encoder_steering(self.bundle(), layer, rec["direction"], alpha, float(rec.get("projection_std") or 1.0), norm_preserve=bool(self.cfg["steering"]["norm_preserve"]))
            preds["N6_GLOBAL_ENCODER"] = decode_manifest(
                self.bundle(),
                target,
                self.cfg,
                system="N6_GLOBAL_ENCODER",
                config_id=f"N6_GLOBAL_ENCODER_L{layer}_a{alpha}",
                out_path=self.root / "predictions" / "N6_GLOBAL_ENCODER.parquet",
                language=self.cfg["decoding"].get("language"),
                hook_builder=global_builder,
                batch_size=int(self.cfg["steering"].get("batch_size", 4)),
                resume=True,
                overwrite=False,
            )
        tables = []
        summaries = []
        for system, pred in preds.items():
            table, summary = self._score_steering_system(pois, target, pred, system)
            tables.append(table.assign(system=system))
            summaries.append(summary)
        all_runs = pd.DataFrame(summaries)
        table_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        runs_path = atomic_write_parquet(all_runs, self.root / "metrics" / "n6_all_runs.parquet")
        atomic_write_parquet(table_df, self.root / "metrics" / "n6_per_example.parquet")
        summary_path = atomic_write_json_file({"label": "preliminary natural pseudo-aligned feasibility result", "layer": layer, "alpha": alpha, "systems": summaries}, self.root / "metrics" / "n6_summary.json")
        body = [
            "N6 is a small held-out steering/control comparison on natural consensus-union pseudo-masks.",
            "",
            "Label: preliminary natural pseudo-aligned feasibility result.",
            "",
            all_runs.to_markdown(index=False) if len(all_runs) else "No held-out steering runs were decoded.",
        ]
        out_report = report(self.root / "reports" / "n6_steering_check.md", "NAT5H N6 Held-out Steering Check", "\n".join(body))
        self.write_status(stage=stage, state=state, timer=timer, output_paths=[str(wrong_path), str(correct_path), str(runs_path), str(summary_path), str(out_report)], gate_evidence={"wrong": len(wrong), "correct": len(correct), "systems": list(preds)}, next_permitted_stage="n7_summary", reason=reason)
        return state

    # ------------------------------------------------------------------
    # N7 final summary
    # ------------------------------------------------------------------
    def n7_summary(self) -> str:
        stage = "n7_summary"
        timer = StageTimer()
        statuses = {}
        for st in STAGES[:-1]:
            statuses[st] = read_status(self.root, st) or {"state": "not_started"}
        conclusion = "E. technical blocker prevented evaluation."
        go_state = "alignment not evaluated"
        if statuses.get("n2_consensus", {}).get("state") in {"completed_no_go", "insufficient_data"}:
            conclusion = "D. insufficient high-confidence natural pseudo-alignments."
            go_state = "alignment insufficient; steering not available"
        elif statuses.get("n1_align_smoke", {}).get("state") == "completed_no_go":
            conclusion = "D. insufficient high-confidence natural pseudo-alignments."
            go_state = "N1 insufficient; N2/N3-N6 not available"
        elif statuses.get("n3_direction_mini", {}).get("state") == "completed_no_go":
            conclusion = "C. no stable direction found."
            go_state = "alignment available; direction not stable"
        elif statuses.get("n4_availability_mini", {}).get("state") == "completed_no_go":
            conclusion = "C. no stable direction found."
            go_state = "direction not sufficiently available on held-out speakers"
        elif statuses.get("n6_steering_check", {}).get("state") == "passed":
            n6 = json.loads((self.root / "metrics" / "n6_summary.json").read_text(encoding="utf-8"))
            local = next((s for s in n6.get("systems", []) if s.get("system") == "N6_LOCAL_CORRECT"), {})
            if local.get("raw_corrected_count", 0) > local.get("raw_corrupted_count", 0):
                conclusion = "A. preliminary direction and steering signal found."
                go_state = "alignment, direction, and steering pilot completed"
            else:
                conclusion = "B. language direction available but steering not corrective."
                go_state = "alignment and direction available; steering not corrective"
        elif statuses.get("n4_availability_mini", {}).get("state") == "passed":
            conclusion = "B. language direction available but steering not corrective."
            go_state = "representation pilot completed; local steering not completed"

        sections = [
            "# NAT5H R2 Mentor Summary",
            "",
            "This is an exploratory natural-audio feasibility run. It does not mark production E1, E2, E3, or E4 as passed.",
            "",
            "## Motivation and question",
            "",
            "Production E1 failed because the existing Whisper-DTW and independent CTC boundary estimates disagreed too strongly for production-grade boundary claims. NAT5H asks a narrower question: whether natural CS-Dialogue utterances with high-confidence automatic multi-aligner agreement can expose a preliminary EN-minus-ZH encoder direction and a small steering signal within one five-hour H100 allocation.",
            "",
            "## Stage statuses",
            "",
            "| stage | state | reason |",
            "| --- | --- | --- |",
        ]
        for st, payload in statuses.items():
            sections.append(f"| {st} | {payload.get('state')} | {payload.get('failure_or_no_go_reason') or ''} |")
        sections.extend(
            [
                "",
                "## Production E1 failure context",
                "",
                "- synthetic boundaries within 100 ms: 0.0600",
                "- synthetic absolute systematic bias: 548.6 ms",
                "- CTC-vs-DTW within 100 ms: 0.0703",
                "- CTC-vs-DTW median absolute disagreement: 370 ms",
                "",
                "NAT5H does not reinterpret those production gates; it uses natural pseudo-alignments only for exploratory feasibility.",
                "",
                "## Natural consensus method",
                "",
                "All timestamps are converted to sample indices relative to sample 0 of the exact short_wav waveform passed to Whisper. No VAD crop, silence trim, full-dialogue timestamp, or synthetic splice coordinate is used.",
                "",
                "A span is accepted only when at least two mechanically valid aligners agree within 200 ms at both start and end. The local steering mask is a consensus-union pseudo-mask, not a gold/oracle mask.",
                "",
                "## Availability state",
                "",
                f"- Scheduler/process success is separate from scientific go/no-go.",
                f"- Current go/no-go state: {go_state}.",
                f"- Alignment schema: `{ALIGNMENT_SCHEMA_VERSION}`.",
                "",
                "## Reports and tables",
                "",
                "- `reports/n1_align_smoke.md`",
                "- `reports/n2_consensus_report.md`",
                "- `reports/n3_direction_mini.md`",
                "- `reports/n4_availability_mini.md`",
                "- `reports/n5_steering_select.md`",
                "- `reports/n6_steering_check.md`",
                "- `metrics/n6_all_runs.parquet`",
                "",
                "## Limitations",
                "",
                "- automatic consensus is not ground truth;",
                "- no human boundary verification is used;",
                "- dev_confirm and official test are intentionally unused;",
                "- held-out steering sample sizes are small and mentor-facing, not production statistical evidence.",
                "",
                "## Conclusion",
                "",
                conclusion,
                "",
            ]
        )
        out = self.root / "reports" / "NAT5H_R2_MENTOR_SUMMARY.md"
        atomic_write_text(out, "\n".join(sections))
        atomic_write_text(self.root / "reports" / "NAT5H_MENTOR_SUMMARY.md", "\n".join(sections))
        self.write_status(stage=stage, state="passed", timer=timer, output_paths=[str(out)], gate_evidence={"conclusion": conclusion}, next_permitted_stage=None)
        return "passed"

    def qwen_smoke(self) -> str:
        stage = "qwen_smoke"
        timer = StageTimer()
        manifest = self.select_smoke_manifest().head(2)
        geometry = self.geometry()
        self.release_bundle()
        qwen = Qwen3ForcedAlignerAdapter(self.cfg)
        payload: dict[str, Any] = {
            "stage": stage,
            "alignment_schema_version": ALIGNMENT_SCHEMA_VERSION,
            "utterances": manifest["utterance_id"].tolist() if len(manifest) else [],
            "language_diagnostics": [],
        }
        rows = []
        state = "passed"
        reason = None
        try:
            languages = qwen.supported_languages()
            for lang in languages:
                try:
                    payload["language_diagnostics"].append(qwen.diagnostic_smoke(manifest.head(1), language=lang))
                    rows.append(qwen.run(manifest, geometry, language=lang, identity=self.identity, diagnostics_dir=self.root / "diagnostics"))
                except Exception as exc:
                    payload["language_diagnostics"].append({"language": lang, "error": repr(exc), "traceback": traceback.format_exc()})
            if rows:
                candidates = pd.concat(rows, ignore_index=True)
                cand_path = atomic_write_parquet(candidates, self.root / "metrics" / "qwen_smoke_candidates.parquet")
                payload["candidate_rows"] = int(len(candidates))
                payload["valid_rows"] = int(candidates["is_valid"].sum()) if len(candidates) else 0
            else:
                cand_path = None
                payload["candidate_rows"] = 0
                payload["valid_rows"] = 0
            if payload["valid_rows"] == 0:
                state = "completed_no_go"
                reason = "qwen_no_valid_rows_in_tiny_smoke"
        except Exception as exc:
            state = "technical_failed"
            reason = repr(exc)
            payload["error"] = reason
            payload["traceback"] = traceback.format_exc()
        finally:
            qwen.unload()
        out_json = atomic_write_json_file(payload, self.root / "diagnostics" / "qwen_adapter_smoke.json")
        out_report = report(
            self.root / "reports" / "qwen_smoke.md",
            "NAT5H R2 Qwen Adapter Smoke",
            "\n".join([
                f"- state: `{state}`",
                f"- reason: `{reason}`" if reason else "- reason: none",
                f"- candidate rows: {payload.get('candidate_rows', 0)}",
                f"- valid rows: {payload.get('valid_rows', 0)}",
            ]),
        )
        self.write_status(
            stage=stage,
            state=state,
            timer=timer,
            output_paths=[str(out_json), str(out_report)] + ([str(cand_path)] if cand_path else []),
            gate_evidence=payload,
            next_permitted_stage="n1_align_smoke" if state in {"passed", "completed_no_go"} else "n7_summary",
            reason=reason,
        )
        return state

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def run(self, overwrite: bool = False) -> int:
        stage_funcs = {
            "n0_preflight": self.n0_preflight,
            "n1_align_smoke": self.n1_align_smoke,
            "n2_consensus": self.n2_consensus,
            "n3_direction_mini": self.n3_direction_mini,
            "n4_availability_mini": self.n4_availability_mini,
            "n5_steering_select": self.n5_steering_select,
            "n6_steering_check": self.n6_steering_check,
            "n7_summary": self.n7_summary,
        }
        permitted = list(STAGES)
        if self.mode == "align_only":
            permitted = STAGES[:3] + ["n7_summary"]
        elif self.mode == "resume_after_alignment":
            n2 = read_status(self.root, "n2_consensus")
            if not n2 or n2.get("state") != "passed":
                raise RuntimeError("resume_after_alignment refused: n2_consensus is not passed")
            n2_candidates = self.root / "metrics" / "n2_consensus_spans.parquet"
            if not n2_candidates.exists():
                raise RuntimeError("resume_after_alignment refused: n2 consensus spans missing")
            passed_spans = pd.read_parquet(n2_candidates)
            if "schema_version" not in passed_spans or set(passed_spans["schema_version"].astype(str)) != {ALIGNMENT_SCHEMA_VERSION}:
                raise RuntimeError("resume_after_alignment refused: n2 spans are not schema v2")
            permitted = STAGES[3:]
        for stage in permitted:
            if stage != "n7_summary" and self.should_stop_before_large_stage():
                timer = StageTimer()
                self.write_status(stage=stage, state="partial_completed", timer=timer, reason="walltime_or_signal_stop_before_stage", next_permitted_stage=stage)
                self.n7_summary()
                return 0
            if self.skip_if_terminal(stage, overwrite):
                continue
            # Respect no-go gates.
            if stage in {"n2_consensus", "n3_direction_mini", "n4_availability_mini", "n5_steering_select", "n6_steering_check"}:
                prev_idx = STAGES.index(stage) - 1
                prev_state = read_status(self.root, STAGES[prev_idx])
                if not prev_state or prev_state.get("state") != "passed":
                    if stage != "n7_summary":
                        continue
            state = stage_funcs[stage]()
            if state == "technical_failed":
                self.n7_summary()
                return 1
            if state in {"completed_no_go", "insufficient_data", "blocked"} and stage != "n7_summary":
                self.n7_summary()
                return 0
        if not (self.root / "status" / "n7_summary.json").exists():
            self.n7_summary()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "align_only", "resume", "qwen_smoke", "resume_after_alignment"])
    parser.add_argument("--config", default="configs/nat5h.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cache = cfg["model_cache"]
    os.environ.setdefault("HF_HOME", cache["hf_home"])
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache["hf_hub_cache"])
    os.environ.setdefault("TORCH_HOME", cache["torch_home"])

    mode = "run" if args.command == "resume" else args.command
    pipe = Nat5HPipeline(cfg, mode=mode)
    if args.command == "qwen_smoke":
        state = pipe.qwen_smoke()
        pipe.n7_summary()
        return 1 if state == "technical_failed" else 0
    return pipe.run(overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
