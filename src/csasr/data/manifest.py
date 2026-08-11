"""Build and load the CS-Dialogue utterance manifest (guide section 4.4).

The local release provides `index/short_wav/{train,dev,test}/{text,wav.scp}` and
`index/total_information/Information_Index.txt` (speaker metadata + CN/EN/MIX
tag). Speaker identity is encoded in the utterance id:

    ZH-CN_U0001_S0_4
    ^^^^^^^^^^^^^^ conversation ZH-CN_U0001, speaker channel S0, segment 4

Utterance ids are globally unique, and no speaker appears in two official
splits, which is what makes the official split speaker-independent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf

from ..utils.hashing import sha256_file
from ..utils.logging import get_logger
from .language_tags import EN, ZH, tag_unit
from .normalize import normalize_and_segment

log = get_logger(__name__)

MANIFEST_COLUMNS = [
    "utterance_id", "conversation_id", "speaker_id", "audio_path", "sample_rate",
    "duration_sec", "official_split", "internal_split", "transcript_raw",
    "transcript_normalized", "contains_en", "contains_zh", "contains_code_switch",
    "dataset_tag", "gender", "age", "region", "device", "topic", "audio_sha256",
]


def parse_utterance_id(utt_id: str) -> tuple[str, str]:
    """Return (conversation_id, speaker_id) for a CS-Dialogue utterance id."""
    parts = utt_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"unexpected utterance id: {utt_id!r}")
    conversation_id = "_".join(parts[:2])          # ZH-CN_U0001
    speaker_id = "_".join(parts[:3])               # ZH-CN_U0001_S0
    return conversation_id, speaker_id


def _read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, _, value = line.partition(" ")
            out[key] = value.strip()
    return out


def _read_information_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["utterance_id"] = df["WAVE ID"].str.replace(".wav", "", regex=False)
    keep = {
        "Gender": "gender", "Age": "age", "Region": "region",
        "Device": "device", "Topic": "topic", "TAG": "dataset_tag",
    }
    cols = ["utterance_id"] + [c for c in keep if c in df.columns]
    out = df[cols].rename(columns=keep)
    return out.drop_duplicates("utterance_id")


def build_manifest(cfg: dict, splits: Iterable[str] | None = None,
                   hash_audio: bool = True, probe_audio: bool = True) -> pd.DataFrame:
    """Construct the utterance manifest from the on-disk index files."""
    dcfg = cfg["data"]
    index_root = Path(dcfg["index_root"])
    audio_root = Path(dcfg["audio_root"])
    splits = list(splits or dcfg["splits"])
    info = _read_information_index(Path(dcfg["information_index"]))

    rows = []
    for split in splits:
        text = _read_kv(index_root / split / "text")
        scp = _read_kv(index_root / split / "wav.scp")
        missing = set(text) ^ set(scp)
        if missing:
            log.warning("%s: %d ids not present in both text and wav.scp", split, len(missing))
        for utt_id in sorted(set(text) & set(scp)):
            conversation_id, speaker_id = parse_utterance_id(utt_id)
            audio_path = audio_root / scp[utt_id]
            raw = text[utt_id]
            norm, units = normalize_and_segment(raw)
            tags = [tag_unit(u) for u in units]
            rows.append(
                {
                    "utterance_id": utt_id,
                    "conversation_id": conversation_id,
                    "speaker_id": speaker_id,
                    "audio_path": str(audio_path),
                    "sample_rate": dcfg["sample_rate"],
                    "duration_sec": float("nan"),
                    "official_split": split,
                    "internal_split": "",
                    "transcript_raw": raw,
                    "transcript_normalized": norm,
                    "contains_en": EN in tags,
                    "contains_zh": ZH in tags,
                    "contains_code_switch": (EN in tags) and (ZH in tags),
                    "audio_sha256": "",
                }
            )

    df = pd.DataFrame(rows)
    df = df.merge(info, on="utterance_id", how="left")
    for col in ("gender", "age", "region", "device", "topic", "dataset_tag"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    if probe_audio:
        durations, srs = [], []
        for p in df["audio_path"]:
            try:
                inf = sf.info(p)
                durations.append(float(inf.duration))
                srs.append(int(inf.samplerate))
            except Exception as exc:  # missing/corrupt audio must be visible
                log.error("cannot read audio %s: %s", p, exc)
                durations.append(float("nan"))
                srs.append(-1)
        df["duration_sec"] = durations
        df["sample_rate"] = srs

    if hash_audio:
        nbytes = int(cfg["data"].get("audio_hash_bytes", 1 << 20))
        df["audio_sha256"] = [
            sha256_file(p, max_bytes=nbytes) if Path(p).exists() else ""
            for p in df["audio_path"]
        ]

    return df[MANIFEST_COLUMNS]


def validate_manifest(df: pd.DataFrame, prohibit_test: bool = True) -> dict:
    """Structural checks required before any experiment runs."""
    report: dict = {}
    report["num_utterances"] = int(len(df))
    report["duplicate_ids"] = int(df["utterance_id"].duplicated().sum())
    assert report["duplicate_ids"] == 0, "duplicate utterance ids in manifest"

    missing_audio = [p for p in df["audio_path"] if not Path(p).exists()]
    report["missing_audio"] = len(missing_audio)
    report["missing_audio_examples"] = missing_audio[:5]

    report["bad_duration"] = int(df["duration_sec"].isna().sum())
    report["bad_sample_rate"] = int((df["sample_rate"] != 16000).sum())
    assert report["missing_audio"] == 0, f"missing audio files: {missing_audio[:5]}"
    assert report["bad_duration"] == 0, "manifest contains unreadable/unknown audio durations"
    assert report["bad_sample_rate"] == 0, "manifest contains non-16-kHz audio"

    # speaker disjointness across official splits
    spk = df.groupby("official_split")["speaker_id"].apply(set).to_dict()
    overlaps = {}
    names = sorted(spk)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = spk[a] & spk[b]
            if inter:
                overlaps[f"{a}|{b}"] = sorted(inter)[:10]
    report["official_split_speaker_overlap"] = overlaps
    assert not overlaps, f"official splits share speakers: {overlaps}"

    report["per_split"] = {
        s: {
            "utterances": int((df["official_split"] == s).sum()),
            "speakers": int(df.loc[df["official_split"] == s, "speaker_id"].nunique()),
            "hours": round(float(df.loc[df["official_split"] == s, "duration_sec"].sum()) / 3600, 2),
            "code_switch_utterances": int(df.loc[df["official_split"] == s, "contains_code_switch"].sum()),
        }
        for s in sorted(df["official_split"].unique())
    }
    if prohibit_test:
        report["test_split_locked"] = True
    return report


def load_manifest(cfg: dict, prohibit_test: bool | None = None) -> pd.DataFrame:
    """Load the manifest, dropping the official test split unless unlocked."""
    path = Path(cfg["data"]["manifest"])
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found at {path}; run `python -m csasr.experiments.p0_baseline "
            f"--build-manifest` or the pipeline stage `manifest` first."
        )
    df = pd.read_parquet(path)
    prohibit = cfg["experiment"]["prohibit_test_split"] if prohibit_test is None else prohibit_test
    if prohibit:
        df = df[df["official_split"] != "test"].reset_index(drop=True)
    return df


def assert_no_test_data(df: pd.DataFrame) -> None:
    """Mechanical block: refuse to proceed if test-split rows are present."""
    if "official_split" in df.columns and (df["official_split"] == "test").any():
        raise RuntimeError(
            "official test-split utterances reached an E1-E5 code path; this is "
            "prohibited by the execution guide (section 2, constraint 6)."
        )
