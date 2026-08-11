"""Why the CTC aligner returns nothing for some reference units.

`existing_ctc` is MMS-FA: a 31-symbol romanized-alphabet CTC model driven by
uroman (`csasr.data.ctc_alignment`). Two mechanisms can silently drop a unit:

* characters absent from the 31-symbol vocabulary are skipped outright
  (`align_units` does `vocab.get(ch)` and continues on None);
* a unit whose Viterbi states are all absorbed by its neighbours yields no span
  at all -- the exposure is highest for a one-character Han unit that romanizes
  into several letters.

This module measures both on real surfaces, on CPU, without loading the model.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ...data.language_tags import tag_unit


def probe_uroman(surfaces: Sequence[str],
                 vocab: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Romanize each surface and record expansion and out-of-vocabulary loss."""
    from ...data.ctc_alignment import romanize

    rows: list[dict[str, Any]] = []
    for surface in surfaces:
        text = str(surface)
        try:
            roman = romanize(text)
        except Exception as exc:                               # pragma: no cover
            rows.append({"surface": text, "romanized": "", "error": repr(exc),
                         "n_chars_in": len(text), "n_chars_out": 0,
                         "expansion_ratio": float("nan"), "oov_chars": -1,
                         "kept_chars": 0, "empty_after_vocab": True,
                         "language": tag_unit(text)})
            continue
        letters = [c for c in roman.lower() if not c.isspace()]
        oov = 0 if vocab is None else sum(1 for c in letters if c not in vocab)
        kept = len(letters) - oov
        rows.append({
            "surface": text,
            "romanized": roman,
            "error": "",
            "language": tag_unit(text),
            "n_chars_in": len(text.replace(" ", "")),
            "n_chars_out": len(letters),
            "expansion_ratio": (len(letters) / max(1, len(text.replace(" ", "")))),
            "oov_chars": int(oov),
            "kept_chars": int(kept),
            "empty_after_vocab": bool(kept == 0),
        })
    return pd.DataFrame(rows)


def characterize_mapping_failures(candidates: pd.DataFrame,
                                  vocab: Mapping[str, int] | None = None,
                                  *, family: str = "existing_ctc",
                                  other_family: str = "whisper_dtw") -> pd.DataFrame:
    """Assign every `mapping_failed` unit a cause bucket.

    Buckets: `empty_after_vocab` (nothing left to align), `single_char_expansion`
    (a one-character unit that romanizes to several letters, the case most
    exposed to neighbour absorption), `short_duration` (the other family gives
    it under 100 ms), or `unexplained`.
    """
    if not len(candidates):
        return pd.DataFrame()
    failed = candidates[(candidates["aligner_family"] == family)
                        & (candidates.get("failure_code", "") == "mapping_failed")]
    if not len(failed):
        return pd.DataFrame()

    other = candidates[(candidates["aligner_family"] == other_family)]
    durations: dict[tuple[str, int], float] = {}
    for _, row in other.iterrows():
        if not bool(row.get("is_valid", True)):
            continue
        durations[(row["utterance_id"], int(row["reference_unit_index"]))] = \
            (float(row["end_sample"]) - float(row["start_sample"])) / 16.0  # ms at 16 kHz

    probe = probe_uroman(failed["reference_text"].astype(str).tolist(), vocab)
    probe = probe.set_index(probe.index)

    rows: list[dict[str, Any]] = []
    for (_, row), (_, p) in zip(failed.iterrows(), probe.iterrows()):
        duration = durations.get((row["utterance_id"], int(row["reference_unit_index"])))
        if p["empty_after_vocab"]:
            cause = "empty_after_vocab"
        elif int(p["n_chars_in"]) == 1 and float(p["expansion_ratio"]) > 1.5:
            cause = "single_char_expansion"
        elif duration is not None and duration < 100.0:
            cause = "short_duration"
        else:
            cause = "unexplained"
        rows.append({
            "utterance_id": row["utterance_id"],
            "unit_id": int(row["reference_unit_index"]),
            "surface": row.get("reference_text", ""),
            "language": row.get("reference_language", ""),
            "romanized": p["romanized"],
            "n_chars_in": int(p["n_chars_in"]),
            "n_chars_out": int(p["n_chars_out"]),
            "expansion_ratio": float(p["expansion_ratio"]),
            "oov_chars": int(p["oov_chars"]),
            "other_family_duration_ms": duration if duration is not None else float("nan"),
            "cause": cause,
        })
    return pd.DataFrame(rows)


def romanization_summary(probe: pd.DataFrame) -> dict[str, Any]:
    if not len(probe):
        return {"n": 0}
    by_language = probe.groupby("language").agg(
        n=("surface", "size"),
        median_expansion=("expansion_ratio", "median"),
        oov_rate=("oov_chars", lambda s: float((s > 0).mean())),
        empty_rate=("empty_after_vocab", "mean")).reset_index()
    return {
        "n": int(len(probe)),
        "median_expansion_ratio": float(probe["expansion_ratio"].median()),
        "fraction_with_oov": float((probe["oov_chars"] > 0).mean()),
        "fraction_empty_after_vocab": float(probe["empty_after_vocab"].mean()),
        "by_language": by_language.to_dict(orient="records"),
    }


def cause_counts(characterized: pd.DataFrame) -> dict[str, int]:
    if not len(characterized):
        return {}
    return {str(k): int(v) for k, v in characterized["cause"].value_counts().items()}
