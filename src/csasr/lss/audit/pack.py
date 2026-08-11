"""Build a blinded, stratified audit pack.

Design decisions that differ from the E1 audit pack:

* the audited object is a **unit edge**, not a switch midpoint. Steering masks
  and localizer labels consume unit edges; nothing downstream uses a midpoint,
  and E1 audited the thing nothing uses;
* items are **blinded**: no aligner family, family count, confidence bin or
  disagreement value reaches the annotator, and the order is shuffled, so a
  verdict cannot be anchored by knowing which system produced the boundary;
* 10% of items are deliberately displaced by a known offset, and 10% are
  duplicated. Without the decoys an audit has no measurable power; without the
  duplicates there is no intra-annotator consistency estimate;
* corrected times are **required** for a defined subset. E1's template left them
  optional, which is why its median/p90 criteria could never have been evaluated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ANNOTATOR_COLUMNS = ("audit_item_id", "annotator_id", "verdict",
                     "corrected_start_sec", "corrected_end_sec", "confidence", "notes")
VERDICTS = ("usable", "boundary_off", "wrong_unit", "unintelligible")


@dataclass(frozen=True)
class AuditPackSpec:
    n_en: int = 100
    n_zh: int = 100
    context_sec: float = 1.5
    duplicate_fraction: float = 0.10
    perturbed_fraction: float = 0.10
    perturb_ms: float = 250.0
    require_corrected_times: int = 120
    strata: tuple[str, ...] = ("language", "confidence_bin")

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "AuditPackSpec":
        block = dict(cfg or {})
        return cls(
            n_en=int(block.get("n_en", cls.n_en)),
            n_zh=int(block.get("n_zh", cls.n_zh)),
            context_sec=float(block.get("context_sec", cls.context_sec)),
            duplicate_fraction=float(block.get("duplicate_fraction", cls.duplicate_fraction)),
            perturbed_fraction=float(block.get("perturbed_fraction", cls.perturbed_fraction)),
            perturb_ms=float(block.get("perturb_ms", cls.perturb_ms)),
            require_corrected_times=int(block.get("require_corrected_times",
                                                  cls.require_corrected_times)),
            strata=tuple(block.get("strata", cls.strata)),
        )


def item_id(salt: str, utterance_id: str, unit_id: int) -> str:
    payload = f"{salt}\0{utterance_id}\0{int(unit_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sample_audit_units(spans: pd.DataFrame, spec: AuditPackSpec, *, seed: int,
                       roles: Sequence[str] | None = None) -> pd.DataFrame:
    """Stratified class-balanced sample: equal English and Mandarin."""
    frame = spans
    if roles and "role" in frame:
        frame = frame[frame["role"].isin(list(roles))]
    if not len(frame):
        return frame

    rng = np.random.default_rng(int(seed))
    picked: list[pd.DataFrame] = []
    for language, target in (("EN", spec.n_en), ("ZH", spec.n_zh)):
        pool = frame[frame["language"] == language]
        if not len(pool):
            continue
        strata = [c for c in spec.strata if c in pool.columns and c != "language"]
        if strata:
            groups = list(pool.groupby(strata, observed=True))
        else:
            groups = [((), pool)]
        per_group = max(1, target // max(1, len(groups)))
        for _, group in groups:
            take = min(per_group, len(group))
            idx = rng.choice(len(group), size=take, replace=False)
            picked.append(group.iloc[np.sort(idx)])
        chosen = pd.concat(picked[-len(groups):]) if groups else pool.head(0)
        if len(chosen) > target:
            keep = rng.choice(len(chosen), size=target, replace=False)
            picked[-len(groups):] = [chosen.iloc[np.sort(keep)]]
    if not picked:
        return frame.head(0)
    return pd.concat(picked).drop_duplicates(["utterance_id", "unit_id"]).reset_index(drop=True)


def build_pack(spans: pd.DataFrame, manifest: pd.DataFrame, out_dir: Path,
               spec: AuditPackSpec, *, seed: int, blinding_seed: int,
               perturb_seed: int, sample_rate: int = 16000,
               render_audio: bool = True) -> dict[str, Any]:
    """Write items.jsonl, the blinding key, a verdict template and the guide."""
    out_dir = Path(out_dir)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    sampled = sample_audit_units(spans, spec, seed=seed)
    if not len(sampled):
        return {"items": 0, "reason": "no spans available to audit"}

    rng = np.random.default_rng(int(perturb_seed))
    n_perturb = int(round(len(sampled) * spec.perturbed_fraction))
    n_duplicate = int(round(len(sampled) * spec.duplicate_fraction))
    perturbed_idx = set(rng.choice(len(sampled), size=min(n_perturb, len(sampled)),
                                   replace=False).tolist())
    duplicate_idx = rng.choice(len(sampled), size=min(n_duplicate, len(sampled)),
                               replace=False).tolist()

    paths = manifest.set_index("utterance_id")["audio_path"].to_dict()
    salt = str(blinding_seed)
    items: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []

    order = list(range(len(sampled))) + list(duplicate_idx)
    shuffle = np.random.default_rng(int(blinding_seed))
    shuffle.shuffle(order)

    for presentation, i in enumerate(order):
        row = sampled.iloc[i]
        is_duplicate = order.index(i) != presentation
        offset_ms = 0.0
        if i in perturbed_idx:
            offset_ms = float(spec.perturb_ms) * (1.0 if presentation % 2 else -1.0)
        start = float(row["consensus_start_sample"]) + offset_ms / 1000.0 * sample_rate
        end = float(row["consensus_end_sample"]) + offset_ms / 1000.0 * sample_rate
        suffix = f"_dup{presentation}" if is_duplicate else ""
        aid = item_id(salt + suffix, str(row["utterance_id"]), int(row["unit_id"]))

        clip_path = out_dir / "clips" / f"{aid}.wav"
        clip_offset = max(0.0, start / sample_rate - spec.context_sec)
        if render_audio:
            _render_clip(paths.get(row["utterance_id"]), clip_path,
                         clip_offset, end / sample_rate + spec.context_sec, sample_rate)

        items.append({
            "audit_item_id": aid,
            "clip_path": str(clip_path),
            "clip_offset_sec": clip_offset,
            # times are relative to the clip, so the annotator never sees the
            # utterance-level coordinates the aligners produced
            "displayed_start_sec": start / sample_rate - clip_offset,
            "displayed_end_sec": end / sample_rate - clip_offset,
            "reference_surface": str(row.get("surface", "")),
            "transcript": "",
            "instructions": "mark the true start and end of the highlighted unit",
        })
        key.append({
            "audit_item_id": aid,
            "utterance_id": str(row["utterance_id"]),
            "unit_id": int(row["unit_id"]),
            "language": str(row.get("language", "")),
            "confidence_bin": str(row.get("confidence_bin", "")),
            "n_families": int(row.get("n_families", 0) or 0),
            "true_start_sec": float(row["consensus_start_sample"]) / sample_rate,
            "true_end_sec": float(row["consensus_end_sample"]) / sample_rate,
            "clip_offset_sec": clip_offset,
            "applied_offset_ms": offset_ms,
            "is_decoy": bool(offset_ms != 0.0),
            "is_duplicate": bool(is_duplicate),
        })

    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "blinding_key.json").write_text(
        json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(columns=list(ANNOTATOR_COLUMNS)).to_csv(
        out_dir / "verdict_template.csv", index=False)
    (out_dir / "ANNOTATION_GUIDE.md").write_text(annotation_guide(spec), encoding="utf-8")

    return {
        "items": len(items),
        "unique_units": int(len(sampled)),
        "decoys": int(sum(1 for k in key if k["is_decoy"])),
        "duplicates": int(sum(1 for k in key if k["is_duplicate"])),
        "en": int((sampled["language"] == "EN").sum()),
        "zh": int((sampled["language"] == "ZH").sum()),
        "blinded_fields": sorted(items[0].keys()) if items else [],
    }


def _render_clip(audio_path: str | None, out_path: Path, start_sec: float,
                 end_sec: float, sample_rate: int) -> None:
    if not audio_path:
        return
    try:
        import soundfile as sf

        from ...models.whisper import load_audio

        audio = load_audio(str(audio_path), sample_rate)
        a = max(0, int(start_sec * sample_rate))
        b = min(len(audio), int(end_sec * sample_rate))
        if b > a:
            sf.write(str(out_path), audio[a:b], sample_rate)
    except Exception:                                          # pragma: no cover
        pass


def annotation_guide(spec: AuditPackSpec) -> str:
    return f"""# Boundary audit: annotation guide

You are marking **where a word actually starts and ends** in a short audio clip.
You are not judging any system: the clips carry no information about which
aligner produced the boundary, and some of them have been deliberately shifted.

## Task

For each item you are given a clip and a proposed start/end time inside it.

1. Listen to the clip.
2. Decide the verdict:
   - `usable` - the proposed boundary is within about 50 ms of where the unit
     really starts and ends;
   - `boundary_off` - the right unit, but the boundary is clearly wrong;
   - `wrong_unit` - the region does not contain the written unit at all;
   - `unintelligible` - you cannot tell, even after repeated listening.
3. **Always fill `corrected_start_sec` and `corrected_end_sec`** unless the
   verdict is `unintelligible`. At least {spec.require_corrected_times} items
   must carry corrected times; without them the boundary-error statistics that
   the gate depends on cannot be computed at all.
4. `confidence` is 1 (unsure) to 5 (certain).

## Conventions

- Times are **relative to the clip**, in seconds.
- The start is the first instant of the unit's own acoustics, not the preceding
  silence or the neighbouring word's release.
- For Mandarin, one item is one character. For English, one item is one word.
- If a unit is preceded by a filled pause, exclude the pause.

## Quality control

- About {int(spec.perturbed_fraction * 100)}% of items are deliberately shifted
  by roughly {spec.perturb_ms:.0f} ms. Marking those `usable` is how the audit
  measures whether it has the power to detect an error at all.
- About {int(spec.duplicate_fraction * 100)}% of items appear twice under
  different ids, to estimate your own consistency.
- At least two annotators score an overlapping subset; disagreements and any
  gap over 100 ms go to adjudication.

## Output

Fill `verdict_template.csv` and save it as
`verdicts_raw/<your-annotator-id>.csv`. Do not edit any other file.
"""
