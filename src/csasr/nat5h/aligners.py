"""NAT5H alignment adapters with one canonical timestamp schema."""
from __future__ import annotations

import gc
import importlib.metadata as im
import json
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from csasr.data.alignment import align_batch, alignment_to_rows
from csasr.data.ctc_alignment import align_units, load_ctc_aligner
from csasr.models.whisper import load_audio

from .coordinates import (
    EncoderGeometry,
)
from .schema import RunIdentity, candidate_row, collapse_duplicate_candidates, validate_candidate_sequence
from .units import ReferenceUnit, build_reference_units


def _audio_num_samples(path: str | Path, target_sr: int = 16000) -> tuple[int, dict[str, Any]]:
    import soundfile as sf

    info = sf.info(str(path))
    if int(info.samplerate) == int(target_sr):
        return int(info.frames), {
            "original_sample_rate": int(info.samplerate),
            "original_num_samples": int(info.frames),
            "resampled_for_whisper": False,
        }
    num = int(round(int(info.frames) * int(target_sr) / int(info.samplerate)))
    return num, {
        "original_sample_rate": int(info.samplerate),
        "original_num_samples": int(info.frames),
        "resampled_for_whisper": True,
    }


def _manifest_unit_map(row: pd.Series) -> list[ReferenceUnit]:
    return build_reference_units(str(row["utterance_id"]), row.get("transcript_raw", ""))


def _content_units(row: pd.Series) -> list[ReferenceUnit]:
    return [u for u in _manifest_unit_map(row) if u.language in {"EN", "ZH"}]


def _candidate(
    *,
    manifest_row: pd.Series,
    unit: ReferenceUnit,
    aligner_family: str,
    aligner_variant: str,
    start_sec: float | None,
    end_sec: float | None,
    geometry: EncoderGeometry,
    identity: RunIdentity,
    source_token_count: int = 1,
    mapping_method: str,
    model_id: str | None = None,
    model_revision: str | None = None,
    is_valid: bool | None = None,
    failure_code: str = "",
    failure_detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict:
    audio_path = manifest_row.get("audio_path", "")
    _, audio_meta = _audio_num_samples(audio_path, geometry.sample_rate)
    meta = {
        **(metadata or {}),
        **audio_meta,
        "normalization_hash": unit.normalization_hash,
        "normalization_version": unit.normalization_version,
        "encoder_step_sec": geometry.encoder_step_sec,
    }
    return candidate_row(
        manifest_row=manifest_row,
        unit=unit,
        aligner_family=aligner_family,
        aligner_variant=aligner_variant,
        start_sec=start_sec,
        end_sec=end_sec,
        geometry=geometry,
        identity=identity,
        source_token_count=source_token_count,
        mapping_method=mapping_method,
        model_id=model_id,
        model_revision=model_revision,
        is_valid=is_valid,
        failure_code=failure_code,
        failure_detail=failure_detail,
        metadata=meta,
    )


def reference_units_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        rows.extend(u.to_dict() for u in _manifest_unit_map(row))
    return pd.DataFrame(rows)


#: `align_batch`'s historical convention: the query that *predicts* token k. The
#: variant label below is unsuffixed for this value so every artifact recorded
#: before the convention was swept keeps its name.
DEFAULT_PRED_START_OFFSET = -1


def dtw_variant_label(pred_start_offset: int = DEFAULT_PRED_START_OFFSET) -> str:
    """Variant name for a decoder-query convention.

    The convention has to be in the name. Two candidate tables produced under
    different offsets are different estimators, and a shared label would let a
    consensus vote silently mix them -- while still counting as one independence
    class, which is correct, so the label is the only place the difference can be
    recorded.
    """
    if int(pred_start_offset) == DEFAULT_PRED_START_OFFSET:
        return "whisper_dtw/zh_median7"
    return f"whisper_dtw/zh_median7_pred{int(pred_start_offset)}"


def selected_pred_start_offset(cfg: dict) -> int:
    """The convention this run aligns with.

    `alignment.dtw_selected.pred_start_offset` is written by the L1a development
    sweep's frozen selection. Absent, the historical default is used, which is
    what every earlier recorded artifact was produced under.
    """
    selected = ((cfg.get("alignment") or {}).get("dtw_selected") or {})
    value = selected.get("pred_start_offset")
    return DEFAULT_PRED_START_OFFSET if value is None else int(value)


def run_whisper_dtw(bundle, manifest: pd.DataFrame, cfg: dict, identity: RunIdentity,
                    *, pred_start_offset: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the existing Whisper cross-attention DTW aligner and convert schema.

    ``pred_start_offset`` overrides the configured decoder-query convention; it
    is what the L1a development sweep varies. Omitted, the value comes from
    `alignment.dtw_selected` and falls back to the historical default, so nothing
    recorded before the sweep existed changes.
    """
    geometry = EncoderGeometry.from_bundle(bundle)
    offset = int(selected_pred_start_offset(cfg) if pred_start_offset is None
                 else pred_start_offset)
    variant = dtw_variant_label(offset)
    rows: list[dict] = []
    mapping_summary: dict[str, Any] = {
        "aligner_family": "whisper_dtw",
        "aligner_variant": variant,
        "pred_start_offset": offset,
        "expected_reference_units": 0,
        "raw_whisper_unit_rows": 0,
        "mapped_unit_rows": 0,
        "unique_mapped_reference_units": 0,
        "unmapped_units": 0,
        "duplicate_mappings_collapsed": 0,
        "valid_canonical_spans": 0,
        "invalid_canonical_spans": 0,
    }
    batch_size = int(cfg.get("alignment", {}).get("batch_size", 4))
    language = cfg.get("alignment", {}).get("language_token", "zh")
    median_filter_width = int(cfg.get("alignment", {}).get("median_filter_width", 7))
    max_text_tokens = int(cfg.get("alignment", {}).get("max_text_tokens", 220))
    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start : start + batch_size].reset_index(drop=True)
        results = align_batch(
            bundle,
            batch,
            language=language,
            median_filter_width=median_filter_width,
            max_text_tokens=max_text_tokens,
            pred_start_offset=offset,
        )
        by_utt = {str(r["utterance_id"]): r for _, r in batch.iterrows()}
        for result in results:
            unit_rows = alignment_to_rows(bundle, result)
            units = _manifest_unit_map(by_utt[str(result.utterance_id)])
            unit_map = {u.unit_id: u for u in units}
            mapping_summary["expected_reference_units"] += sum(1 for u in units if u.language in {"EN", "ZH"})
            mapping_summary["raw_whisper_unit_rows"] += len(unit_rows)
            for ur in unit_rows:
                unit = unit_map.get(int(ur["unit_id"]))
                if unit is None:
                    mapping_summary["unmapped_units"] += 1
                    continue
                if unit.language not in {"EN", "ZH"}:
                    # Preserve OTHER in mapping diagnostics, but do not make it
                    # a scientific alignment candidate for consensus.
                    mapping_summary["unmapped_units"] += 1
                    continue
                rows.append(_candidate(
                    manifest_row=by_utt[str(result.utterance_id)],
                    unit=unit,
                    aligner_family="whisper_dtw",
                    aligner_variant=variant,
                    start_sec=float(ur["start_sec"]),
                    end_sec=float(ur["end_sec"]),
                    geometry=geometry,
                    identity=identity,
                    source_token_count=1,
                    mapping_method="existing_reference_unit_dtw",
                    model_id=bundle.model_id,
                    model_revision=bundle.revision,
                    metadata={
                        "alignment_source": ur.get("alignment_source"),
                        "origin_verified": "converted_from_short_wav_dtw_seconds",
                        "alignment_confidence": float(ur.get("alignment_confidence", np.nan)),
                        "pred_start_offset": offset,
                    },
                ))
                mapping_summary["mapped_unit_rows"] += 1
    out = pd.DataFrame(rows)
    out, dup = collapse_duplicate_candidates(out)
    mapping_summary.update(dup)
    out = validate_candidate_sequence(out)
    if len(out):
        mapping_summary["unique_mapped_reference_units"] = int(out[["utterance_id", "reference_unit_index"]].drop_duplicates().shape[0])
        mapping_summary["valid_canonical_spans"] = int(out["is_valid"].sum())
        mapping_summary["invalid_canonical_spans"] = int((~out["is_valid"]).sum())
        coverage = mapping_summary["unique_mapped_reference_units"] / max(mapping_summary["expected_reference_units"], 1)
        if coverage > 1.0:
            raise AssertionError(f"DTW coverage exceeded 1 after schema-v2 canonicalization: {coverage}")
        mapping_summary["unit_coverage"] = float(coverage)
        mapping_summary["failure_counts"] = out.loc[~out["is_valid"], "failure_code"].value_counts().to_dict()
    return out, mapping_summary


def ctc_available(cfg: dict) -> tuple[bool, str | None]:
    try:
        load_ctc_aligner(cfg)
        return True, None
    except Exception as exc:
        return False, repr(exc)


def run_existing_ctc(manifest: pd.DataFrame, cfg: dict, geometry: EncoderGeometry, identity: RunIdentity) -> pd.DataFrame:
    aligner = load_ctc_aligner(cfg)
    rows: list[dict] = []
    for _, row in manifest.iterrows():
        units = _manifest_unit_map(row)
        surfaces = [u.surface for u in units]
        content = {u.unit_id: u for u in units if u.language in {"EN", "ZH"}}
        try:
            audio = load_audio(row["audio_path"], geometry.sample_rate)
            spans = align_units(aligner, audio, surfaces, geometry.sample_rate)
        except Exception as exc:
            for unit in content.values():
                rows.append(_candidate(
                    manifest_row=row,
                    unit=unit,
                    aligner_family="existing_ctc",
                    aligner_variant="existing_ctc/default",
                    start_sec=None,
                    end_sec=None,
                    geometry=geometry,
                    identity=identity,
                    source_token_count=0,
                    mapping_method="ctc_viterbi_reference_unit",
                    model_id=cfg.get("alignment", {}).get("ctc", {}).get("model_id"),
                    is_valid=False,
                    failure_code="aligner_exception",
                    failure_detail=repr(exc),
                    metadata={"ctc_exception": repr(exc)},
                ))
            continue
        unit_map = {u.unit_id: u for u in units}
        seen = set()
        for sp in spans:
            unit = unit_map.get(int(sp["unit_id"]))
            if unit is None or unit.language not in {"EN", "ZH"}:
                continue
            seen.add(unit.unit_id)
            rows.append(_candidate(
                manifest_row=row,
                unit=unit,
                aligner_family="existing_ctc",
                aligner_variant="existing_ctc/default",
                start_sec=float(sp["start_sec"]),
                end_sec=float(sp["end_sec"]),
                geometry=geometry,
                identity=identity,
                source_token_count=1,
                mapping_method="ctc_viterbi_reference_unit",
                model_id=cfg.get("alignment", {}).get("ctc", {}).get("model_id"),
                metadata={
                    "ctc_model_id": cfg.get("alignment", {}).get("ctc", {}).get("model_id"),
                    "ctc_frame_sec": getattr(aligner, "frame_sec", None),
                    "ctc_package": "transformers",
                },
            ))
        for unit_id, unit in content.items():
            if unit_id not in seen:
                rows.append(_candidate(
                    manifest_row=row,
                    unit=unit,
                    aligner_family="existing_ctc",
                    aligner_variant="existing_ctc/default",
                    start_sec=None,
                    end_sec=None,
                    geometry=geometry,
                    identity=identity,
                    source_token_count=0,
                    mapping_method="ctc_viterbi_reference_unit",
                    model_id=cfg.get("alignment", {}).get("ctc", {}).get("model_id"),
                    is_valid=False,
                    failure_code="mapping_failed",
                    failure_detail="CTC returned no span for this reference unit",
                ))
    out = pd.DataFrame(rows)
    out, _ = collapse_duplicate_candidates(out)
    return validate_candidate_sequence(out)


def _item_attr(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def qwen_result_structure(result: Any) -> dict[str, Any]:
    outer = result
    first_sample = None
    first_item = None
    outer_len = None
    first_sample_len = None
    try:
        outer_len = len(outer)
    except Exception:
        pass
    try:
        first_sample = outer[0] if outer_len else None
    except Exception:
        first_sample = None
    try:
        first_sample_len = len(first_sample) if first_sample is not None else None
    except Exception:
        pass
    try:
        first_item = first_sample[0] if first_sample_len else None
    except Exception:
        first_item = None
    attrs = []
    if first_item is not None:
        attrs = sorted(a for a in dir(first_item) if not a.startswith("_"))
    return {
        "return_outer_type": type(outer).__name__,
        "outer_length": outer_len,
        "first_sample_type": type(first_sample).__name__ if first_sample is not None else None,
        "first_sample_length": first_sample_len,
        "first_item_type": type(first_item).__name__ if first_item is not None else None,
        "first_item_public_attributes": attrs,
    }


def qwen_items_from_result(result: Any, sample_index: int = 0) -> list[Any]:
    """Parse installed qwen-asr 0.0.6 nested List[ForcedAlignResult] output."""
    if isinstance(result, list):
        if len(result) <= sample_index:
            return []
        sample = result[sample_index]
    else:
        sample = result
    if hasattr(sample, "items"):
        return list(getattr(sample, "items"))
    return list(sample or [])


def _qwen_items_to_unit_spans(items: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        text = str(_item_attr(item, "text", "") or "")
        start = _item_attr(item, "start_time", _item_attr(item, "start", None))
        end = _item_attr(item, "end_time", _item_attr(item, "end", None))
        if start is None or end is None:
            continue
        units = build_reference_units("_qwen_item", text)
        content = units if units else []
        if not content:
            continue
        total = max(float(end) - float(start), 0.0)
        step = total / len(content) if len(content) else 0.0
        for i, unit in enumerate(content):
            s = float(start) + step * i
            e = float(start) + step * (i + 1)
            out.append({"surface": unit.surface, "start_sec": s, "end_sec": e, "raw_text": text})
    return out


class Qwen3ForcedAlignerAdapter:
    def __init__(self, cfg: dict):
        qcfg = cfg["qwen_aligner"]
        self.model_id = qcfg["model_id"]
        self.local_model_dir = qcfg["local_model_dir"]
        self.dtype = qcfg.get("dtype", "bfloat16")
        self.device = qcfg.get("device", "cuda:0")
        self._aligner = None
        self.package_version = im.version("qwen-asr")

    def load(self):
        if self._aligner is None:
            import torch
            from qwen_asr import Qwen3ForcedAligner

            dtype = torch.bfloat16 if str(self.dtype) == "bfloat16" else torch.float32
            self._aligner = Qwen3ForcedAligner.from_pretrained(
                self.local_model_dir,
                dtype=dtype,
                device_map=self.device,
            )
            self._aligner.model.eval()
            for p in self._aligner.model.parameters():
                p.requires_grad_(False)
        return self._aligner

    def unload(self) -> None:
        self._aligner = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def supported_languages(self) -> list[str]:
        aligner = self.load()
        try:
            langs = aligner.get_supported_languages()
        except Exception:
            langs = None
        if langs:
            wanted = []
            for lang in ("Chinese", "English"):
                if lang in langs or lang.lower() in {str(x).lower() for x in langs}:
                    wanted.append(lang)
            return wanted or [str(langs[0])]
        return ["Chinese", "English"]

    def diagnostic_smoke(self, manifest: pd.DataFrame, language: str) -> dict[str, Any]:
        aligner = self.load()
        if manifest.empty:
            return {"language": language, "error": "empty_manifest"}
        row = manifest.iloc[0]
        units = _manifest_unit_map(row)
        normalized_text = units[0].normalized_transcript if units else str(row.get("transcript_raw", ""))
        result = aligner.align(row["audio_path"], normalized_text, language=language)
        items = qwen_items_from_result(result)
        return {
            "installed_package_version": self.package_version,
            "model_id": self.model_id,
            "local_model_dir": self.local_model_dir,
            "language": language,
            **qwen_result_structure(result),
            "parsed_item_count": len(items),
            "utterance_id": row.get("utterance_id"),
            "audio_path": row.get("audio_path"),
            "audio_duration": row.get("duration_sec"),
            "transcript_length": len(normalized_text),
        }

    def run(
        self,
        manifest: pd.DataFrame,
        geometry: EncoderGeometry,
        language: str,
        identity: RunIdentity,
        diagnostics_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        aligner = self.load()
        rows: list[dict] = []
        failures_path = Path(diagnostics_dir) / "qwen_utterance_failures.jsonl" if diagnostics_dir else None
        if failures_path:
            failures_path.parent.mkdir(parents=True, exist_ok=True)
        for _, row in manifest.iterrows():
            units = _manifest_unit_map(row)
            ref_content = [u for u in units if u.language in {"EN", "ZH"}]
            if not ref_content:
                continue
            normalized_text = units[0].normalized_transcript if units else str(row.get("transcript_raw", ""))
            structure: dict[str, Any] = {}
            try:
                result = aligner.align(row["audio_path"], normalized_text, language=language)
                structure = qwen_result_structure(result)
                items = qwen_items_from_result(result)
            except Exception as exc:
                detail = {
                    "utterance_id": row.get("utterance_id"),
                    "audio_path": row.get("audio_path"),
                    "audio_duration": row.get("duration_sec"),
                    "transcript_length": len(normalized_text),
                    "language": language,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if failures_path:
                    with failures_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
                for unit in ref_content:
                    rows.append(_candidate(
                        manifest_row=row,
                        unit=unit,
                        aligner_family="qwen_forced_aligner",
                        aligner_variant=f"qwen_forced_aligner/{language}",
                        start_sec=None,
                        end_sec=None,
                        geometry=geometry,
                        identity=identity,
                        source_token_count=0,
                        mapping_method="qwen_sequential_surface",
                        model_id=self.model_id,
                        is_valid=False,
                        failure_code="aligner_exception",
                        failure_detail=f"{type(exc).__name__}: {exc}",
                        metadata=detail,
                    ))
                continue
            spans = _qwen_items_to_unit_spans(items)
            if not spans:
                detail = {
                    "utterance_id": row.get("utterance_id"),
                    "audio_path": row.get("audio_path"),
                    "audio_duration": row.get("duration_sec"),
                    "transcript_length": len(normalized_text),
                    "language": language,
                    "return_object_structure": structure,
                    "raw_item_count": len(items),
                    "parsed_item_count": 0,
                    "mapping_failure_reason": "empty_aligner_output",
                }
                if failures_path:
                    with failures_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
                for unit in ref_content:
                    rows.append(_candidate(
                        manifest_row=row,
                        unit=unit,
                        aligner_family="qwen_forced_aligner",
                        aligner_variant=f"qwen_forced_aligner/{language}",
                        start_sec=None,
                        end_sec=None,
                        geometry=geometry,
                        identity=identity,
                        source_token_count=0,
                        mapping_method="qwen_sequential_surface",
                        model_id=self.model_id,
                        is_valid=False,
                        failure_code="empty_aligner_output",
                        failure_detail="Qwen returned no parseable items",
                        metadata=detail,
                    ))
                continue
            cursor = 0
            for unit in ref_content:
                matched = None
                for j in range(cursor, len(spans)):
                    if spans[j]["surface"] == unit.surface:
                        matched = spans[j]
                        cursor = j + 1
                        break
                if matched is None:
                    rows.append(_candidate(
                        manifest_row=row,
                        unit=unit,
                        aligner_family="qwen_forced_aligner",
                        aligner_variant=f"qwen_forced_aligner/{language}",
                        start_sec=None,
                        end_sec=None,
                        geometry=geometry,
                        identity=identity,
                        source_token_count=0,
                        mapping_method="qwen_sequential_surface",
                        model_id=self.model_id,
                        is_valid=False,
                        failure_code="mapping_failed",
                        failure_detail=f"no Qwen item matched reference surface {unit.surface!r}",
                        metadata={
                            "qwen_language": language,
                            "qwen_raw_item_count": len(items),
                            "qwen_parsed_span_count": len(spans),
                            "return_object_structure": structure,
                        },
                    ))
                    continue
                rows.append(_candidate(
                    manifest_row=row,
                    unit=unit,
                    aligner_family="qwen_forced_aligner",
                    aligner_variant=f"qwen_forced_aligner/{language}",
                    start_sec=float(matched["start_sec"]),
                    end_sec=float(matched["end_sec"]),
                    geometry=geometry,
                    identity=identity,
                    source_token_count=1,
                    mapping_method="qwen_sequential_surface",
                    model_id=self.model_id,
                    metadata={
                        "qwen_model_id": self.model_id,
                        "qwen_local_model_dir": self.local_model_dir,
                        "qwen_language": language,
                        "qwen_package_version": self.package_version,
                        "qwen_raw_item_text": matched.get("raw_text"),
                        "return_object_structure": structure,
                    },
                ))
        out = pd.DataFrame(rows)
        out, _ = collapse_duplicate_candidates(out)
        return validate_candidate_sequence(out)
