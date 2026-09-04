"""Split loading, evaluation population, and target construction.

`load_split("D-test")` raises. It is meant to. `D-dev-confirm` loads only for
the human-approved Confirm step and refuses otherwise.

The evaluation population is the frozen v2r3 development sample -- 300
utterances of `D-dev-select` over 20 dialogues -- and the target set is the
489 baseline-error embedded-English units at the 400 ms tier. Both are rebuilt
through the existing frozen code (`v2r3_day5_expansion._spans_at_floor`,
`v2r3_oracle_screen.build_targets`), never re-derived here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as C


class LockedSplitError(RuntimeError):
    """Raised on any attempt to read a locked split."""


class UnauthorizedSplitError(RuntimeError):
    """Raised when a held-out split is requested without explicit authority."""


# ---------------------------------------------------------------------------
# split loading
# ---------------------------------------------------------------------------

def _role_root(cfg: dict) -> Path:
    return Path(cfg["v2_namespace"]["role_root"])


def load_split(name: str, cfg: dict, *, confirm_authorized: bool = False) -> pd.DataFrame:
    """Load one role manifest.

    `D-test` is locked and always raises: it tunes nothing and this harness has
    no reason to open it. `D-dev-confirm` is the single final development
    confirmation and requires explicit authorisation from the caller, which the
    orchestrator only grants after a human approves the printed configuration.
    """
    if name in C.LOCKED_SPLITS:
        raise LockedSplitError(
            f"{name} is LOCKED. It tunes nothing and is never read by this "
            f"harness. If a future protocol authorises it, that decision is "
            f"made by a human and recorded, not by relaxing this loader.")
    if name == C.DEV_CONFIRM and not confirm_authorized:
        raise UnauthorizedSplitError(
            f"{name} is the one final development confirmation and has NEVER "
            f"been opened. Track A prints its single selected configuration "
            f"and stops; a human authorises the Confirm run explicitly.")
    path = _role_root(cfg) / f"role_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"role manifest missing: {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# the frozen development population
# ---------------------------------------------------------------------------

@dataclass
class Population:
    """The evaluation population and everything derived from it."""

    manifest: pd.DataFrame          # 300 utterances, D-dev-select
    targets: pd.DataFrame           # 489 rows, one per baseline-error EN unit
    poi: pd.DataFrame               # per-POI baseline status
    baseline_hypotheses: dict[str, str]
    split: str
    counts: dict[str, Any]

    @property
    def utterance_ids(self) -> list[str]:
        return [str(u) for u in self.manifest["utterance_id"]]

    def subsample(self, n: int) -> "Population":
        """Smoke population: the first `n` utterances that carry a target."""
        with_targets = list(dict.fromkeys(str(u) for u in self.targets["utterance_id"]))
        keep = set(with_targets[:n])
        if len(keep) < n:
            extra = [u for u in self.utterance_ids if u not in keep]
            keep |= set(extra[: n - len(keep)])
        manifest = self.manifest[self.manifest["utterance_id"].isin(keep)].reset_index(drop=True)
        targets = self.targets[self.targets["utterance_id"].isin(keep)].reset_index(drop=True)
        poi = self.poi[self.poi["utterance_id"].isin(keep)].reset_index(drop=True)
        return Population(
            manifest=manifest, targets=targets, poi=poi,
            baseline_hypotheses={k: v for k, v in self.baseline_hypotheses.items()
                                 if k in keep},
            split=self.split,
            counts={"utterances": int(len(manifest)),
                    "dialogues": int(manifest["dialogue_id"].nunique()),
                    "targets": int(len(targets)), "smoke": True},
        )


def _paths(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["experiment"]["output_root"])
    return {
        "candidates": root / "alignments" / "candidates_all.parquet",
        "role_root": _role_root(cfg),
        "poi_root": root.parent.parent / "baselines" / "generation_001",
    }


@lru_cache(maxsize=4)
def _cached_population(split: str, config_path: str, confirm_authorized: bool):
    from csasr.utils.config import load_config
    cfg = load_config(config_path)
    return _build_population(cfg, split, confirm_authorized=confirm_authorized)


def _build_population(cfg: dict, split: str, *, confirm_authorized: bool) -> Population:
    from csasr.experiments import v2r3_day5_expansion as D5

    paths = _paths(cfg)
    manifest = load_split(split, cfg, confirm_authorized=confirm_authorized)
    candidates = pd.read_parquet(paths["candidates"])
    candidates = candidates[candidates["role"].astype(str) == split]
    if candidates.empty:
        raise RuntimeError(f"no candidate rows for role {split}")

    # the 300-utterance frozen sample is exactly the utterances that carry
    # candidate rows; the role manifest itself holds every utterance of the
    # dialogue, which is not the evaluated population
    sampled = set(candidates["utterance_id"].astype(str))
    manifest = manifest[manifest["utterance_id"].astype(str).isin(sampled)].reset_index(drop=True)

    poi_path = paths["poi_root"] / f"poi_{split}.parquet"
    poi = pd.read_parquet(poi_path).rename(columns={"poi_index": "reference_unit_index"})

    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))
    spans = D5._spans_at_floor(candidates, dialogue_of, poi, C.TARGET_FLOOR_MS)
    spans = spans[spans["role"] == split].reset_index(drop=True)

    counts = {
        "split": split,
        "utterances": int(len(manifest)),
        "dialogues": int(manifest["dialogue_id"].nunique()),
        "spans": int(len(spans)),
        "floor_ms": C.TARGET_FLOOR_MS,
    }
    return Population(manifest=manifest, targets=spans, poi=poi,
                      baseline_hypotheses={}, split=split, counts=counts)


def build_population(bundle, cfg: dict, split: str = C.DEV_SELECT, *,
                     confirm_authorized: bool = False,
                     assert_anchors: bool = True) -> Population:
    """The evaluated population with its 489-row target table.

    `bundle` is needed because target construction tokenises the reference to
    find the decoder step of each unit; it is the frozen v2r3 routine.
    """
    from csasr.experiments import v2r3_oracle_screen as OS

    pop = _build_population(cfg, split, confirm_authorized=confirm_authorized)
    targets = OS.build_targets(bundle, None, pop.targets, pop.poi, pop.manifest)
    pop.targets = targets
    pop.counts["targets"] = int(len(targets))
    pop.counts["utterances_with_targets"] = int(targets["utterance_id"].nunique())
    if assert_anchors and split == C.DEV_SELECT:
        assert_population_anchors(pop)
    return pop


def assert_population_anchors(pop: Population) -> None:
    """Fail loudly if the population is not the one every prior number is on."""
    checks = [
        ("utterances", pop.counts["utterances"], C.ANCHOR_DEV_SELECT_UTTERANCES),
        ("dialogues", pop.counts["dialogues"], C.ANCHOR_DEV_SELECT_DIALOGUES),
        ("targets", pop.counts["targets"], C.ANCHOR_TARGETS),
        ("utterances_with_targets", pop.counts["utterances_with_targets"],
         C.ANCHOR_DEV_SELECT_TARGET_UTTERANCES),
    ]
    bad = [(n, o, e) for n, o, e in checks if int(o) != int(e)]
    if bad:
        lines = "\n".join(f"  {n}: observed {o}, expected {e}" for n, o, e in bad)
        raise AssertionError(
            "population does not match the frozen v2r3 sample every prior "
            f"result is measured on:\n{lines}\n"
            "Continuing would produce numbers that cannot be compared to "
            "anything. Stopping.")


# ---------------------------------------------------------------------------
# unit-level reference tables used by the gates
# ---------------------------------------------------------------------------

def unit_table(pop: Population, hypotheses: dict[str, str]) -> pd.DataFrame:
    """Per reference unit: language tag and correctness under `hypotheses`.

    This is the denominator machinery for G4 (ZH retention, 5,711) and for
    corruption over baseline-correct units (6,264). Both are recomputed with
    current source rather than read from the pre-repair stored fields.
    """
    from csasr.data.language_tags import EN, ZH, tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.evaluation.pier import unit_status

    rows: list[dict[str, Any]] = []
    for _, row in pop.manifest.iterrows():
        utterance = str(row["utterance_id"])
        hypothesis = hypotheses.get(utterance)
        if hypothesis is None:
            continue
        reference = str(row["transcript_raw"])
        _, units = normalize_and_segment(reference)
        status = unit_status(reference, hypothesis)
        for index, unit in enumerate(units):
            tag = tag_unit(unit)
            rows.append({
                "utterance_id": utterance,
                "dialogue_id": str(row["dialogue_id"]),
                "reference_unit_index": int(index),
                "surface": unit.surface,
                "language": tag,
                "is_en": tag == EN,
                "is_zh": tag == ZH,
                "correct": bool(status.get(index, (False, ""))[0]),
                "category": str(status.get(index, (False, ""))[1]),
            })
    return pd.DataFrame(rows)


def duration_of(pop: Population) -> dict[str, float]:
    return {str(u): float(d) for u, d in
            zip(pop.manifest["utterance_id"], pop.manifest["duration_sec"])}


def audio_of(pop: Population) -> dict[str, str]:
    return {str(u): str(p) for u, p in
            zip(pop.manifest["utterance_id"], pop.manifest["audio_path"])}


def reference_of(pop: Population) -> dict[str, str]:
    return {str(u): str(t) for u, t in
            zip(pop.manifest["utterance_id"], pop.manifest["transcript_raw"])}
