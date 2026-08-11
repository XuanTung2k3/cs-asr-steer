"""Internal split construction (guide section 4.2 / 4.3).

Official dev speakers are divided into `dev_select` (~20) and `dev_confirm`
(~10) deterministically; training utterances are subsetted for direction
construction. Nothing here ever touches the official test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .manifest import assert_no_test_data

log = get_logger(__name__)

INTERNAL_SPLITS = ("train", "dev_select", "dev_confirm")


def split_dev_speakers(dev_speakers: list[str], n_select: int = 20,
                       seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministic speaker-level partition of the official dev split."""
    ordered = sorted(set(dev_speakers))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    shuffled = [ordered[i] for i in perm]
    select = sorted(shuffled[:n_select])
    confirm = sorted(shuffled[n_select:])
    assert not (set(select) & set(confirm)), "dev_select and dev_confirm overlap"
    return select, confirm


def assign_internal_splits(df: pd.DataFrame, n_select: int = 20,
                           seed: int = 42) -> pd.DataFrame:
    assert_no_test_data(df)
    df = df.copy()
    dev_speakers = sorted(df.loc[df["official_split"] == "dev", "speaker_id"].unique())
    select, confirm = split_dev_speakers(dev_speakers, n_select=n_select, seed=seed)
    mapping = {s: "dev_select" for s in select}
    mapping.update({s: "dev_confirm" for s in confirm})
    df["internal_split"] = [
        "train" if sp == "train" else mapping.get(spk, "")
        for sp, spk in zip(df["official_split"], df["speaker_id"])
    ]
    return df


def eligible_for_direction(df: pd.DataFrame) -> pd.Series:
    """Utterances that can contribute to direction construction.

    Frame-level eligibility (minimum EN/ZH frames, alignment quality) is
    enforced later in E2; this is the transcript-level pre-filter.
    """
    return (
        df["contains_code_switch"]
        & df["duration_sec"].notna()
        & (df["duration_sec"] > 0.3)
        & (df["duration_sec"] <= 30.0)
        & (df["transcript_normalized"].str.len() > 0)
    )


def build_direction_subsets(df: pd.DataFrame, pilot_size: int = 5000,
                            seed: int = 42,
                            bootstrap_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
                            ) -> dict[str, pd.DataFrame]:
    """Create pilot/full direction manifests plus per-seed subsample manifests.

    The pilot samples speakers approximately evenly so that a few talkative
    speakers cannot dominate the direction estimate.
    """
    train = df[(df["official_split"] == "train") & eligible_for_direction(df)].copy()
    full = train.sort_values("utterance_id").reset_index(drop=True)

    speakers = sorted(full["speaker_id"].unique())
    per_speaker = max(1, int(np.ceil(pilot_size / max(1, len(speakers)))))
    rng = np.random.default_rng(seed)
    picks = []
    for spk in speakers:
        rows = full[full["speaker_id"] == spk]
        take = min(per_speaker, len(rows))
        idx = rng.choice(len(rows), size=take, replace=False)
        picks.append(rows.iloc[np.sort(idx)])
    pilot = pd.concat(picks).sort_values("utterance_id").reset_index(drop=True)
    if len(pilot) > pilot_size:
        rng2 = np.random.default_rng(seed + 1)
        keep = np.sort(rng2.choice(len(pilot), size=pilot_size, replace=False))
        pilot = pilot.iloc[keep].reset_index(drop=True)

    subsets = {"train_direction_pilot": pilot, "train_direction_full": full}
    for s in bootstrap_seeds:
        r = np.random.default_rng(s)
        size = min(len(pilot), max(1, int(0.8 * len(pilot))))
        idx = np.sort(r.choice(len(pilot), size=size, replace=False))
        subsets[f"train_direction_seed{s}"] = pilot.iloc[idx].reset_index(drop=True)
    return subsets


def validate_splits(df: pd.DataFrame, subsets: dict[str, pd.DataFrame]) -> dict:
    """Assert speaker disjointness and report split statistics."""
    report: dict = {}
    groups = {
        name: set(df.loc[df["internal_split"] == name, "speaker_id"])
        for name in INTERNAL_SPLITS
    }
    for a, b in (("train", "dev_select"), ("train", "dev_confirm"),
                 ("dev_select", "dev_confirm")):
        inter = groups[a] & groups[b]
        report[f"overlap_{a}_{b}"] = sorted(inter)
        assert not inter, f"speaker leakage between {a} and {b}: {sorted(inter)[:5]}"

    for name in INTERNAL_SPLITS:
        sub = df[df["internal_split"] == name]
        report[name] = {
            "speakers": int(sub["speaker_id"].nunique()),
            "utterances": int(len(sub)),
            "hours": round(float(sub["duration_sec"].sum()) / 3600, 2),
            "code_switch_utterances": int(sub["contains_code_switch"].sum()),
        }
    for name, sub in subsets.items():
        dev_spk = groups["dev_select"] | groups["dev_confirm"]
        leaked = set(sub["speaker_id"]) & dev_spk
        assert not leaked, f"{name} contains development speakers: {sorted(leaked)[:5]}"
        report[name] = {
            "utterances": int(len(sub)),
            "speakers": int(sub["speaker_id"].nunique()),
            "hours": round(float(sub["duration_sec"].sum()) / 3600, 2),
        }
    return report


def write_split_manifests(out_dir: str | Path, df: pd.DataFrame,
                          subsets: dict[str, pd.DataFrame], report: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in INTERNAL_SPLITS:
        sub = df[df["internal_split"] == name]
        if len(sub):
            sub.to_parquet(out_dir / f"{name}.parquet", index=False)
    for name, sub in subsets.items():
        sub.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_subset(cfg: dict, name: str) -> pd.DataFrame:
    """Load a named subset manifest (dev_select, train_direction_pilot, ...)."""
    from ..utils.config import artifacts_root

    path = artifacts_root(cfg) / "manifests" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"subset manifest missing: {path}")
    df = pd.read_parquet(path)
    assert_no_test_data(df)
    return df
