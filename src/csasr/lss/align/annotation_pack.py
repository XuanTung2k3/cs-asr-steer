"""Export a pack of code-switch boundary clips for human annotation.

The pack exists to obtain a lexical reference that is **independent of every
aligner**.  A provisional machine boundary is therefore used for exactly one
purpose -- deciding which three seconds of audio to cut -- and appears nowhere
the annotator can see it: the TextGrid tiers ship empty, no tier is named after
a predicted time, and no clip carries a mark at the centre.

Nothing in this module estimates, rounds, copies, or otherwise writes a boundary
value into an annotation tier.  ``write_textgrid`` cannot do so: it accepts tier
names only, and always emits one empty interval spanning the whole clip.

Read-only against cached candidates; every write lands under a caller-supplied
output directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

#: The only role this pack may draw from.
SOURCE_ROLE = "D-dev-select"
#: Provisional boundaries come from this family, for centring only.
PROVISIONAL_FAMILY = "existing_ctc"
#: Mandarin is the matrix language of CS-Dialogue; English spans are embedded.
MATRIX_LANGUAGE = "ZH"
EMBEDDED_LANGUAGE = "EN"

TIER_NAMES = ("matrix_end", "embedded_start")
CLIP_DURATION_SEC = 3.0
CLIP_SAMPLE_RATE = 16000
STRATA_LABELS = ("short", "medium", "long")
TAINT = "development_only_diagnostic"
MANIFEST_SCHEMA = "lss_cs_boundary_annotation_pack_v1"


class AnnotationPackError(RuntimeError):
    """Raised when the pack cannot be built from the requested source."""


def _canonical(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def fingerprint(item: dict[str, Any]) -> str:
    """Stable identifier for one clip, from its identity and its cut.

    Deliberately includes the clip geometry: a pack cut differently is a
    different pack, and annotations must never be matched across the two.
    """
    payload = {
        "utterance_id": str(item["utterance_id"]),
        "left_reference_unit_index": int(item["left_reference_unit_index"]),
        "right_reference_unit_index": int(item["right_reference_unit_index"]),
        "switch_direction": str(item["switch_direction"]),
        "clip_start_offset_sec": round(float(item["clip_start_offset_sec"]), 6),
        "clip_duration_sec": round(float(item["clip_duration_sec"]), 6),
        "sample_rate": int(item["sample_rate"]),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _stream_key(label: str) -> int:
    """Process-stable integer seed component derived from a stratum name."""
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _language_runs(group: pd.DataFrame) -> list[dict[str, Any]]:
    """Consecutive same-language unit runs, in reference order."""
    ordered = group.sort_values("reference_unit_index")
    runs: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        language = str(row["reference_language"])
        if runs and runs[-1]["language"] == language:
            runs[-1]["rows"].append(row)
        else:
            runs.append({"language": language, "rows": [row]})
    return runs


def switch_universe(candidates: pd.DataFrame, *,
                    role: str = SOURCE_ROLE,
                    family: str = PROVISIONAL_FAMILY) -> tuple[pd.DataFrame, dict[str, int]]:
    """Every code-switch boundary eligible for annotation, plus exclusions.

    One row per boundary between two adjacent language runs.  An English run
    contributes its onset (a ZH->EN switch) and its offset (an EN->ZH switch);
    both carry that run's duration, so a tercile of embedded-span duration is
    defined for either direction.
    """
    if "role" not in candidates.columns:
        raise AnnotationPackError("candidates carry no role column; refusing to guess the split")
    rows = candidates[(candidates["role"].astype(str) == str(role))
                      & (candidates["aligner_family"].astype(str) == str(family))]
    if not len(rows):
        raise AnnotationPackError(f"no {family} rows for role {role}")

    excluded = {"non_contiguous_reference_unit_index": 0, "unit_not_valid": 0,
                "non_finite_boundary": 0, "non_positive_embedded_span": 0}
    items: list[dict[str, Any]] = []
    for utterance, group in rows.groupby("utterance_id", sort=True):
        runs = _language_runs(group)
        for position, run in enumerate(runs):
            if run["language"] != EMBEDDED_LANGUAGE:
                continue
            span_start = float(run["rows"][0]["start_sec"])
            span_end = float(run["rows"][-1]["end_sec"])
            span_duration = span_end - span_start
            if not np.isfinite(span_duration) or span_duration <= 0:
                excluded["non_positive_embedded_span"] += 1
                continue
            neighbours = []
            if position > 0 and runs[position - 1]["language"] == MATRIX_LANGUAGE:
                neighbours.append(("ZH->EN", runs[position - 1]["rows"][-1], run["rows"][0]))
            if (position + 1 < len(runs)
                    and runs[position + 1]["language"] == MATRIX_LANGUAGE):
                neighbours.append(("EN->ZH", run["rows"][-1], runs[position + 1]["rows"][0]))
            for direction, left, right in neighbours:
                if int(right["reference_unit_index"]) - int(left["reference_unit_index"]) != 1:
                    excluded["non_contiguous_reference_unit_index"] += 1
                    continue
                if not (bool(left.get("is_valid", True)) and bool(right.get("is_valid", True))):
                    excluded["unit_not_valid"] += 1
                    continue
                left_end, right_start = float(left["end_sec"]), float(right["start_sec"])
                if not (np.isfinite(left_end) and np.isfinite(right_start)):
                    excluded["non_finite_boundary"] += 1
                    continue
                items.append({
                    "utterance_id": str(utterance),
                    "conversation_id": str(left.get("conversation_id", "")),
                    "speaker_id": str(left.get("speaker", "")),
                    "audio_path": str(left.get("audio_path", "")),
                    "audio_duration_sec": float(left.get("audio_duration_sec", float("nan"))),
                    "left_reference_unit_index": int(left["reference_unit_index"]),
                    "right_reference_unit_index": int(right["reference_unit_index"]),
                    "switch_direction": direction,
                    "word_before_switch": str(left.get("surface", "")),
                    "word_after_switch": str(right.get("surface", "")),
                    "language_before_switch": str(left["reference_language"]),
                    "language_after_switch": str(right["reference_language"]),
                    "embedded_span_duration_sec": float(span_duration),
                    # Centring only. Never written into a tier, never shown in
                    # annotator-facing material.
                    "_provisional_center_sec": (left_end + right_start) / 2.0,
                })
    universe = pd.DataFrame(items)
    if len(universe):
        universe = universe.sort_values(
            ["utterance_id", "left_reference_unit_index", "switch_direction"]
        ).reset_index(drop=True)
    return universe, excluded


def duration_terciles(universe: pd.DataFrame) -> tuple[float, float]:
    """Cut points splitting embedded-span duration into three equal parts."""
    values = universe["embedded_span_duration_sec"].astype(float).to_numpy()
    lower, upper = np.quantile(values, [1 / 3, 2 / 3], method="linear")
    return float(lower), float(upper)


def assign_strata(universe: pd.DataFrame,
                  cuts: tuple[float, float]) -> pd.DataFrame:
    lower, upper = cuts
    duration = universe["embedded_span_duration_sec"].astype(float)
    stratum = np.where(duration <= lower, STRATA_LABELS[0],
                       np.where(duration <= upper, STRATA_LABELS[1], STRATA_LABELS[2]))
    frame = universe.copy()
    frame["duration_stratum"] = stratum
    frame["stratum"] = frame["switch_direction"] + "|" + frame["duration_stratum"]
    return frame


def allocate(total: int, keys: Sequence[str],
             capacity: dict[str, int]) -> dict[str, int]:
    """Spread ``total`` as evenly as possible, never exceeding availability.

    Deterministic: the remainder goes to strata in sorted key order, and any
    shortfall in an exhausted stratum is redistributed to the others.
    """
    keys = list(keys)
    quota = {key: 0 for key in keys}
    remaining = int(total)
    while remaining > 0:
        open_keys = [k for k in keys if quota[k] < capacity.get(k, 0)]
        if not open_keys:
            break
        share, extra = divmod(remaining, len(open_keys))
        if share == 0:
            for key in sorted(open_keys)[:extra]:
                quota[key] += 1
                remaining -= 1
            break
        progressed = False
        for key in sorted(open_keys):
            take = min(share, capacity[key] - quota[key])
            quota[key] += take
            remaining -= take
            progressed = progressed or take > 0
        if not progressed:
            break
    return quota


def sample_pack(universe: pd.DataFrame, *, seed: int, n_primary: int,
                n_double: int) -> pd.DataFrame:
    """Deterministically draw primary and double-annotation items per stratum.

    Double-annotation items come from the same strata and are disjoint from the
    primary set, so inter-annotator agreement is measured without any item
    entering the main metrics twice.
    """
    cuts = duration_terciles(universe)
    frame = assign_strata(universe, cuts)
    keys = sorted(frame["stratum"].unique())
    capacity = {key: int((frame["stratum"] == key).sum()) for key in keys}

    total = int(n_primary) + int(n_double)
    combined = allocate(total, keys, capacity)
    primary_quota = allocate(int(n_primary), keys,
                             {key: combined[key] for key in keys})

    picks: list[pd.DataFrame] = []
    for key in keys:
        take = combined[key]
        if take <= 0:
            continue
        stratum = frame[frame["stratum"] == key].sort_values(
            ["utterance_id", "left_reference_unit_index", "switch_direction"]
        ).reset_index(drop=True)
        # A per-stratum generator keeps one stratum's draw independent of how
        # many items another stratum happened to contain.  The stream is keyed
        # by a sha256 digest, not by hash(), which numpy would accept and which
        # is salted per process -- the draw has to survive a new interpreter.
        rng = np.random.default_rng([int(seed), _stream_key(key)])
        order = rng.permutation(len(stratum))[:take]
        # The double-annotation flag follows the *draw* order, not the canonical
        # sort order.  Flagging the tail of a sorted stratum would hand every
        # agreement item to whichever speakers sort last, which is a skewed
        # subsample dressed up as a random one.
        double = set(int(i) for i in order[int(primary_quota[key]):])
        chosen = stratum.iloc[sorted(order)].copy()
        chosen["double_annotation"] = [int(i) in double for i in sorted(order)]
        picks.append(chosen)

    if not picks:
        raise AnnotationPackError("no items could be sampled from the universe")
    pack = pd.concat(picks, ignore_index=True)
    pack = pack.sort_values(
        ["double_annotation", "stratum", "utterance_id", "left_reference_unit_index"]
    ).reset_index(drop=True)
    pack["seed"] = int(seed)
    pack["tercile_lower_sec"], pack["tercile_upper_sec"] = cuts
    return pack


def clip_bounds(center_sec: float, *, duration: float = CLIP_DURATION_SEC
                ) -> tuple[float, float]:
    """Clip start and the boundary's position within the clip.

    The provisional boundary sits at the exact centre of every clip, so the
    offset is returned unrounded and may be negative near the file head; the
    caller pads rather than sliding the window, because sliding would move the
    boundary away from the centre.
    """
    start = float(center_sec) - float(duration) / 2.0
    return start, float(duration) / 2.0


def extract_clip(samples: np.ndarray, sample_rate: int, start_sec: float, *,
                 duration: float = CLIP_DURATION_SEC
                 ) -> tuple[np.ndarray, float, float]:
    """Cut ``duration`` seconds from ``start_sec``, zero-padding past the edges."""
    want = int(round(duration * sample_rate))
    begin = int(round(start_sec * sample_rate))
    end = begin + want
    pad_head = max(0, -begin)
    pad_tail = max(0, end - len(samples))
    body = samples[max(0, begin):min(len(samples), end)]
    clip = np.concatenate([
        np.zeros(pad_head, dtype=samples.dtype), body,
        np.zeros(pad_tail, dtype=samples.dtype)])
    if len(clip) != want:                       # a clip shorter than the file
        clip = clip[:want]
    return clip, pad_head / sample_rate, pad_tail / sample_rate


def textgrid_text(duration: float = CLIP_DURATION_SEC,
                  tiers: Sequence[str] = TIER_NAMES) -> str:
    """A Praat TextGrid whose every tier is one empty, unmarked interval.

    There is no parameter for a boundary time or an interval label, so this
    function cannot emit a pre-filled annotation however it is called.
    """
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
             "xmin = 0 ", f"xmax = {duration} ", "tiers? <exists> ",
             f"size = {len(tiers)} ", "item []: "]
    for index, name in enumerate(tiers, start=1):
        lines += [f"    item [{index}]:",
                  '        class = "IntervalTier" ',
                  f'        name = "{name}" ',
                  "        xmin = 0 ",
                  f"        xmax = {duration} ",
                  "        intervals: size = 1 ",
                  "        intervals [1]:",
                  "            xmin = 0 ",
                  f"            xmax = {duration} ",
                  '            text = "" ']
    return "\n".join(lines) + "\n"


def parse_textgrid_tiers(text: str) -> list[dict[str, Any]]:
    """Read back tier names, interval counts, and labels, for verification."""
    tiers: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("name = "):
            current = {"name": line.split("=", 1)[1].strip().strip('"'),
                       "labels": [], "intervals": 0}
            tiers.append(current)
        elif line.startswith("intervals: size = ") and current is not None:
            current["intervals"] = int(line.split("=")[-1].strip())
        elif line.startswith("text = ") and current is not None:
            current["labels"].append(line.split("=", 1)[1].strip().strip('"'))
    return tiers


ANNOTATOR_README = """# Code-switch boundary annotation pack

Every clip is 3 seconds of conversational Mandarin-English speech containing
exactly one language switch. Your job is to mark two times **in seconds
relative to the start of the clip**.

## What to mark

Each clip ships with a Praat TextGrid containing two empty interval tiers:

- **`matrix_end`** — the lexical end of the Mandarin (matrix-language) word at
  this switch.
- **`embedded_start`** — the lexical start of the English (embedded) word at
  this switch.

The `switch_direction` column in the `items.csv` beside your clips tells you
which word comes first:

- `ZH->EN` — the Mandarin word comes first. `matrix_end` is where that Mandarin
  word stops; `embedded_start` is where the English word begins.
- `EN->ZH` — the English word comes first. `embedded_start` is where that
  English word *begins*, which in this direction is earlier in the clip than
  `matrix_end`, the point where the following Mandarin word starts.

In both directions you are marking the same two things: the edge of the
Mandarin word that touches the switch, and the edge of the English word that
touches the switch. The tier names describe the language, not the order.

## How to mark

- Mark the **lexical** boundary: where the word's own articulation begins or
  ends. Do not include a following pause, breath, or filled hesitation.
- The two marks are allowed to coincide, and they are allowed to be separated
  by silence. Do not force them together and do not force them apart.
- **Always give a best estimate.** If a boundary is genuinely ambiguous —
  overlapping speech, a blended coarticulation, an unclear onset — still place
  your best mark, then record the item as ambiguous in `ambiguity_notes.csv`.
  A flagged best estimate is usable evidence; a skipped item is not.
- Do not skip an item because it sounds difficult.

## What is deliberately absent

The tiers ship empty on purpose. No machine estimate of either boundary is
included anywhere in this pack's clips or TextGrids, because the whole point of
the exercise is to obtain a reference that owes nothing to an aligner. Please do
not consult any other alignment while annotating.

Clips were cut around a provisional machine estimate purely to decide which
three seconds to include. **That estimate is not marked, and you should not try
to infer it.** Treat the clip as three seconds of audio with no privileged
position, and in particular do not assume the switch sits at the midpoint.

## Layout

- `clips/` — the {n_primary} primary items, with an `items.csv` listing each
  clip's switch direction, transcript, and the two flanking words.
- `double_annotation/` — {n_double} items for a second sitting, used to measure
  inter-annotator agreement, with its own `items.csv`. Annotate these
  separately, and do not look at your first pass while doing so.
- `ambiguity_notes.csv` — record ambiguous items here, one row per item. It
  ships with a header and no rows.
- `manifest.csv` / `manifest.json` — **for the analyst, not the annotator.**
  These record each clip's offset within its source recording, from which the
  provisional machine estimate could be reconstructed. Do not open them while
  annotating; everything you need is in `items.csv`.
"""

#: What the annotator may see: identity, language, and words. No timing at all,
#: so nothing here can be inverted into a machine boundary.
ANNOTATOR_COLUMNS = ("item_id", "fingerprint", "clip_path", "textgrid_path",
                     "switch_direction", "word_before_switch",
                     "word_after_switch", "transcript")

AMBIGUITY_HEADER = ("fingerprint", "item_id", "annotator",
                    "ambiguous_tier", "reason")
