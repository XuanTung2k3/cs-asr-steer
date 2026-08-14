"""Build and load the CS-Dialogue utterance manifest (guide section 4.4).

The local release provides `index/short_wav/{train,dev,test}/{text,wav.scp}` and
`index/total_information/Information_Index.txt` (speaker metadata + CN/EN/MIX
tag). Speaker identity is encoded in the utterance id:

    ZH-CN_U0001_S0_4
    ^^^^^^^^^^^^^^ conversation ZH-CN_U0001, speaker channel S0, segment 4

Utterance ids are globally unique, and no speaker appears in two official
splits, which is what makes the official split speaker-independent.

`conversation_id` names one *side* of a dialogue, not a dialogue. CS-Dialogue is
100 two-party conversations recorded by 200 speakers, each participant's channel
stored as its own recording, and the channel token is `S0` for all 38,682
utterances -- so `conversation_id` and `speaker_id` carry identical information
and both identify a single participant. `dialogue_id` is the missing third
level: it groups the two participants of one recorded session. The release ships
no session id, so it is *inferred* from the corpus `Speaker ID` column and must
travel with the evidence `dialogue_pairing_evidence` produces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf

from ..utils.hashing import sha256_file
from ..utils.logging import get_logger
from .language_tags import EN, ZH, tag_unit
from .normalize import normalize_and_segment

log = get_logger(__name__)

MANIFEST_COLUMNS = [
    "utterance_id", "conversation_id", "speaker_id", "dialogue_id",
    "corpus_speaker_id", "audio_path", "sample_rate",
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


class DialoguePairingError(RuntimeError):
    """The corpus speaker numbering cannot support an unambiguous pairing."""


#: Seed and draw count for the permutation nulls reported as evidence for the
#: pairing inference.  Recorded in the payload so every number is reproducible.
PAIRING_NULL_SEED = 20260815
PAIRING_NULL_DRAWS = 2000


def dialogue_id_for(corpus_speaker_id: int | str) -> str:
    """Dialogue label for one participant's corpus `Speaker ID`.

    The two participants of a session are the consecutive speaker numbers
    (2k-1, 2k).  `derive_dialogue_ids` verifies that rather than assuming it.
    """
    number = int(corpus_speaker_id)
    if number < 1:
        raise DialoguePairingError(f"speaker number must be positive, got {number!r}")
    return f"CSD{(number + 1) // 2:04d}"


def derive_dialogue_ids(info: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation: its corpus speaker number and its dialogue.

    Raises `DialoguePairingError` rather than guessing whenever the numbering
    cannot carry the pairing -- a non-integer or absent speaker number, a
    conversation holding more than one speaker number (or the reverse), or a
    dialogue whose partner is missing.

    The speaker numbering is deliberately *not* required to be one contiguous
    run: this release numbers speakers in three series with gaps between them.
    Completeness is therefore checked per pair, not over the global range. Every
    gap must fall *between* pairs; a gap falling inside one would leave a
    half-empty dialogue, which the group-size check rejects.
    """
    if "corpus_speaker_id" not in info.columns:
        raise DialoguePairingError(
            "the information index lacks `corpus_speaker_id`; the corpus "
            "`Speaker ID` column must be retained to derive a dialogue")

    frame = info[["utterance_id", "corpus_speaker_id"]].copy()
    absent = int(frame["corpus_speaker_id"].isna().sum())
    text = frame["corpus_speaker_id"].fillna("").astype(str).str.strip()
    absent += int((text == "").sum())
    if absent:
        raise DialoguePairingError(
            f"{absent} utterance(s) carry no corpus Speaker ID; the information "
            "index does not cover the manifest, so no dialogue can be derived")
    malformed = sorted(set(text[~text.str.fullmatch(r"\d+")]))
    if malformed:
        raise DialoguePairingError(
            f"corpus Speaker ID is not integer-valued for {len(malformed)} "
            f"distinct value(s): {malformed[:5]}")

    frame["corpus_speaker_id"] = text.astype(int)
    frame["conversation_id"] = [parse_utterance_id(u)[0] for u in frame["utterance_id"]]

    per_conversation = frame.groupby("conversation_id")["corpus_speaker_id"].nunique()
    mixed = sorted(per_conversation[per_conversation > 1].index)
    if mixed:
        raise DialoguePairingError(
            f"{len(mixed)} conversation(s) carry more than one speaker number, so "
            f"conversation and speaker are not 1:1: {mixed[:5]}")

    pairs = (frame[["conversation_id", "corpus_speaker_id"]]
             .drop_duplicates().reset_index(drop=True))
    per_speaker = pairs.groupby("corpus_speaker_id")["conversation_id"].nunique()
    shared = [int(s) for s in per_speaker[per_speaker > 1].index]
    if shared:
        raise DialoguePairingError(
            f"{len(shared)} speaker number(s) appear in more than one "
            f"conversation: {sorted(shared)[:5]}")

    pairs["dialogue_id"] = [dialogue_id_for(s) for s in pairs["corpus_speaker_id"]]
    sizes = pairs.groupby("dialogue_id")["corpus_speaker_id"].nunique()
    incomplete = sizes[sizes != 2]
    if len(incomplete):
        detail = {str(k): int(v) for k, v in list(incomplete.items())[:5]}
        raise DialoguePairingError(
            f"{len(incomplete)} of {len(sizes)} dialogue(s) do not have exactly two "
            f"participants: {detail}. The (2k-1, 2k) pairing is not supported by "
            "this speaker numbering; do not guess a session grouping from it.")
    return pairs.sort_values("corpus_speaker_id").reset_index(drop=True)


def _pair_null(values: np.ndarray, *, seed: int, draws: int) -> dict:
    """How often a *random* perfect matching would agree on `values`.

    The pairing is inferred, so every agreement statistic needs the number
    chance alone produces next to it.
    """
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(draws):
        order = rng.permutation(len(values))
        left, right = values[order[0::2]], values[order[1::2]]
        hits.append(int(sum(1 for a, b in zip(left, right) if a == b)))
    return {"mean": round(float(np.mean(hits)), 2), "max": int(max(hits)),
            "draws": int(draws), "seed": int(seed)}


def dialogue_pairing_evidence(frame: pd.DataFrame, *, seed: int = PAIRING_NULL_SEED,
                              draws: int = PAIRING_NULL_DRAWS) -> dict:
    """Evidence that the inferred (2k-1, 2k) grouping is the real dialogue.

    `dialogue_id` is inferred rather than read, so it must never travel without
    this payload.  Each agreement statistic is reported beside the value chance
    alone would produce; a reader who distrusts the inference can judge it
    instead of taking it on faith.
    """
    conversations = (
        frame.groupby("conversation_id")
        .agg(dialogue_id=("dialogue_id", "first"),
             corpus_speaker_id=("corpus_speaker_id", "first"),
             official_split=("official_split", "first"),
             device=("device", "first"),
             gender=("gender", "first"),
             utterances=("utterance_id", "nunique"))
        .reset_index())
    # a conversation covers 2-6 topics, so the comparable unit is the topic set
    topics = frame.groupby("conversation_id")["topic"].apply(
        lambda s: frozenset(str(v) for v in s.unique()))
    conversations["topics"] = conversations["conversation_id"].map(topics)

    grouped = conversations.groupby("dialogue_id")
    sizes = grouped.size()
    both = grouped.agg(list)

    def agree(column: str) -> int:
        return int(sum(1 for v in both[column] if len(v) == 2 and v[0] == v[1]))

    numbers = sorted(conversations["corpus_speaker_id"])
    steps = [(a, b) for a, b in zip(numbers, numbers[1:]) if b - a != 1]
    jaccard = [len(v[0] & v[1]) / len(v[0] | v[1])
               for v in both["topics"] if len(v) == 2]

    payload = {
        "inference": "consecutive corpus Speaker ID numbers (2k-1, 2k) are the "
                     "two participants of one recorded session",
        "conversations": int(len(conversations)),
        "dialogues": int(len(sizes)),

        "check_1_group_sizes": {
            "dialogues": int(len(sizes)),
            "with_exactly_two_members": int((sizes == 2).sum()),
            "size_histogram": {int(k): int(v) for k, v in sizes.value_counts().items()},
            "passes": bool((sizes == 2).all()),
        },
        "check_2_topic_set_agreement": {
            "unit": "set of topics per conversation (topic is not constant within "
                    "a conversation: the release covers 2-6 topics each)",
            "pairs_with_identical_topic_sets": agree("topics"),
            "of_pairs": int(len(sizes)),
            "mean_within_pair_jaccard": round(float(np.mean(jaccard)), 4) if jaccard else None,
            "distinct_topic_sets": int(conversations["topics"].nunique()),
            "chance_null": _pair_null(conversations["topics"].to_numpy(),
                                      seed=seed, draws=draws),
        },
        "check_3_official_split_agreement": {
            "pairs_in_one_official_split": agree("official_split"),
            "of_pairs": int(len(sizes)),
            "chance_null": _pair_null(conversations["official_split"].to_numpy(),
                                      seed=seed, draws=draws),
        },
        "check_4_composition": {
            "device_per_pair": {"/".join(sorted(v)): int(c) for v, c in
                                both["device"].map(lambda v: tuple(sorted(v)))
                                .value_counts().items()},
            "gender_per_pair": {"".join(sorted(v)): int(c) for v, c in
                                both["gender"].map(lambda v: tuple(sorted(v)))
                                .value_counts().items()},
        },
        "check_5_speaker_conversation_one_to_one": {
            "conversations": int(conversations["conversation_id"].nunique()),
            "corpus_speaker_ids": int(conversations["corpus_speaker_id"].nunique()),
            "is_one_to_one": bool(conversations["conversation_id"].nunique()
                                  == conversations["corpus_speaker_id"].nunique()
                                  == len(conversations)),
        },

        "speaker_numbering": {
            "min": int(numbers[0]), "max": int(numbers[-1]), "count": len(numbers),
            "is_one_contiguous_run": numbers == list(range(numbers[0], numbers[0] + len(numbers))),
            "gaps_between_pairs": int(sum(1 for a, _ in steps if a % 2 == 0)),
            "gaps_inside_a_pair": int(sum(1 for a, _ in steps if a % 2 == 1)),
        },
        "dialogues_per_official_split": {
            str(k): int(v) for k, v in
            conversations.groupby("official_split")["dialogue_id"].nunique().items()},
    }
    payload["all_checks_pass"] = bool(
        payload["check_1_group_sizes"]["passes"]
        and payload["check_2_topic_set_agreement"]["pairs_with_identical_topic_sets"]
        == payload["check_2_topic_set_agreement"]["of_pairs"]
        and payload["check_3_official_split_agreement"]["pairs_in_one_official_split"]
        == payload["check_3_official_split_agreement"]["of_pairs"]
        and payload["check_5_speaker_conversation_one_to_one"]["is_one_to_one"]
        and payload["speaker_numbering"]["gaps_inside_a_pair"] == 0)
    return payload


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
        # column 1 of the release: the only speaker identity the corpus states
        # rather than encodes in a filename, and the basis for `dialogue_id`
        "Speaker ID": "corpus_speaker_id",
        "Gender": "gender", "Age": "age", "Region": "region",
        "Device": "device", "Topic": "topic", "TAG": "dataset_tag",
    }
    cols = ["utterance_id"] + [c for c in keep if c in df.columns]
    out = df[cols].rename(columns=keep)
    if "corpus_speaker_id" in out.columns:
        out["corpus_speaker_id"] = out["corpus_speaker_id"].str.strip()
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

    # the third identity level: which two participants shared one session.
    # `pairs` also carries the speaker number as an int, which is what makes
    # `dialogue_id` re-derivable from the manifest alone.
    pairs = derive_dialogue_ids(df)
    df = df.drop(columns=["corpus_speaker_id"]).merge(
        pairs, on="conversation_id", how="left", validate="many_to_one")

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
