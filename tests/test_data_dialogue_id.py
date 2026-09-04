"""`dialogue_id`: the inferred two-party session grouping.

CS-Dialogue stores each participant's channel separately, so `conversation_id`
and `speaker_id` both name one side of a dialogue.  `dialogue_id` groups the two
sides.  Nothing in the release states that grouping, so these tests exercise the
refusals as hard as the happy path: a pairing that is wrong but silent would
merge two independent speakers into one bootstrap cluster and narrow every
interval computed downstream.
"""
from __future__ import annotations

import pandas as pd
import pytest

from csasr.data.manifest import (MANIFEST_COLUMNS, DialoguePairingError,
                                 build_manifest, derive_dialogue_ids,
                                 dialogue_id_for, dialogue_pairing_evidence,
                                 parse_utterance_id)


def _info(speaker_by_conversation: dict[str, str], per_conversation: int = 2,
          topics: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """An information-index frame: one row per utterance."""
    rows = []
    for conversation, speaker in speaker_by_conversation.items():
        topic_list = (topics or {}).get(conversation, ["personal topics"])
        for u in range(per_conversation):
            rows.append({"utterance_id": f"{conversation}_S0_{u}",
                         "corpus_speaker_id": speaker,
                         "topic": topic_list[u % len(topic_list)]})
    return pd.DataFrame(rows)


def _consecutive(n_pairs: int, start: int = 1) -> dict[str, str]:
    """`n_pairs` complete dialogues, numbered consecutively from `start`."""
    return {f"ZH-CN_U{i:04d}": f"{i:04d}"
            for i in range(start, start + 2 * n_pairs)}


# --------------------------------------------------------------------------
# the pairing rule


def test_dialogue_id_pairs_consecutive_speaker_numbers():
    assert dialogue_id_for(1) == dialogue_id_for(2) == "CSD0001"
    assert dialogue_id_for(3) == dialogue_id_for(4) == "CSD0002"
    assert dialogue_id_for("0004") == "CSD0002"          # zero-padded corpus form
    assert dialogue_id_for(2013) == dialogue_id_for(2014) == "CSD1007"
    assert dialogue_id_for(2) != dialogue_id_for(3)      # pairs do not straddle


def test_pairing_is_deterministic_and_order_independent():
    info = _info(_consecutive(4))
    first = derive_dialogue_ids(info)
    again = derive_dialogue_ids(info)
    shuffled = derive_dialogue_ids(info.sample(frac=1.0, random_state=7))
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, shuffled)
    assert first["dialogue_id"].nunique() == 4


def test_gaps_between_pairs_are_accepted():
    """The corpus numbers speakers in three series with gaps between them.

    Requiring one contiguous run would reject the real corpus; what matters is
    that no gap falls *inside* a pair.
    """
    pairs = derive_dialogue_ids(_info({"ZH-CN_U0001": "0001", "ZH-CN_U0002": "0002",
                                       "ZH-CN_U1011": "1011", "ZH-CN_U1012": "1012"}))
    assert pairs["dialogue_id"].nunique() == 2
    assert set(pairs["dialogue_id"]) == {"CSD0001", "CSD0506"}


# --------------------------------------------------------------------------
# refusals: each of these would otherwise mispair silently


def test_missing_partner_raises():
    with pytest.raises(DialoguePairingError, match="exactly two participants"):
        derive_dialogue_ids(_info({"ZH-CN_U0001": "0001", "ZH-CN_U0002": "0002",
                                   "ZH-CN_U0003": "0003"}))


def test_gap_inside_a_pair_raises():
    """`{1,2,4,5}` looks consecutive-ish but splits one dialogue in half."""
    with pytest.raises(DialoguePairingError, match="exactly two participants"):
        derive_dialogue_ids(_info({"ZH-CN_U0001": "0001", "ZH-CN_U0002": "0002",
                                   "ZH-CN_U0004": "0004", "ZH-CN_U0005": "0005"}))


def test_non_integer_speaker_id_raises():
    with pytest.raises(DialoguePairingError, match="not integer-valued"):
        derive_dialogue_ids(_info({"ZH-CN_U0001": "S01", "ZH-CN_U0002": "0002"}))


def test_absent_speaker_id_raises():
    info = _info(_consecutive(1))
    info.loc[0, "corpus_speaker_id"] = ""
    with pytest.raises(DialoguePairingError, match="no corpus Speaker ID"):
        derive_dialogue_ids(info)


def test_discarded_speaker_column_raises():
    info = _info(_consecutive(1)).drop(columns=["corpus_speaker_id"])
    with pytest.raises(DialoguePairingError, match="must be retained"):
        derive_dialogue_ids(info)


def test_conversation_holding_two_speaker_numbers_raises():
    info = _info(_consecutive(1))
    info.loc[info.index[-1], "utterance_id"] = "ZH-CN_U0001_S0_9"   # 0002 into U0001
    with pytest.raises(DialoguePairingError, match="not 1:1"):
        derive_dialogue_ids(info)


def test_speaker_number_in_two_conversations_raises():
    with pytest.raises(DialoguePairingError, match="more than one"):
        derive_dialogue_ids(_info({"ZH-CN_U0001": "0001", "ZH-CN_U0002": "0001"}))


# --------------------------------------------------------------------------
# evidence


def _paired_frame(n_pairs: int = 4, *, split: str = "train") -> pd.DataFrame:
    rows = []
    for i in range(1, 2 * n_pairs + 1):
        conversation = f"ZH-CN_U{i:04d}"
        for u in range(2):
            rows.append({
                "utterance_id": f"{conversation}_S0_{u}",
                "conversation_id": conversation,
                "corpus_speaker_id": i,
                "dialogue_id": dialogue_id_for(i),
                "official_split": split,
                "device": "Android" if i % 2 else "iphone",
                "gender": "F",
                "topic": ["work", "study"][u],
            })
    return pd.DataFrame(rows)


def test_evidence_reports_every_check_with_its_chance_baseline():
    evidence = dialogue_pairing_evidence(_paired_frame(), draws=50)
    assert evidence["all_checks_pass"] is True
    assert evidence["check_1_group_sizes"]["with_exactly_two_members"] == 4
    assert evidence["check_2_topic_set_agreement"]["pairs_with_identical_topic_sets"] == 4
    assert evidence["check_2_topic_set_agreement"]["mean_within_pair_jaccard"] == 1.0
    assert evidence["check_3_official_split_agreement"]["pairs_in_one_official_split"] == 4
    assert evidence["check_5_speaker_conversation_one_to_one"]["is_one_to_one"] is True
    assert evidence["speaker_numbering"]["gaps_inside_a_pair"] == 0
    # a chance baseline must accompany each agreement statistic, seeded
    for check in ("check_2_topic_set_agreement", "check_3_official_split_agreement"):
        null = evidence[check]["chance_null"]
        assert null["draws"] == 50 and "mean" in null and "max" in null


def test_evidence_is_reproducible():
    frame = _paired_frame()
    assert dialogue_pairing_evidence(frame, draws=50) == \
        dialogue_pairing_evidence(frame, draws=50)


def test_evidence_fails_when_a_pair_straddles_official_splits():
    frame = _paired_frame()
    frame.loc[frame["conversation_id"] == "ZH-CN_U0002", "official_split"] = "dev"
    evidence = dialogue_pairing_evidence(frame, draws=50)
    assert evidence["check_3_official_split_agreement"]["pairs_in_one_official_split"] == 3
    assert evidence["all_checks_pass"] is False


def test_evidence_fails_when_topic_sets_disagree():
    frame = _paired_frame()
    frame.loc[frame["conversation_id"] == "ZH-CN_U0002", "topic"] = "unrelated"
    evidence = dialogue_pairing_evidence(frame, draws=50)
    assert evidence["check_2_topic_set_agreement"]["pairs_with_identical_topic_sets"] == 3
    assert evidence["all_checks_pass"] is False


# --------------------------------------------------------------------------
# the manifest itself


def _corpus(tmp_path, n_pairs: int = 3) -> dict:
    """A miniature CS-Dialogue release: index files plus an information index."""
    index_root = tmp_path / "index"
    rows = ["Speaker ID\tGender\tAge\tRegion\tDevice\tTopic\tWAVE ID\tTAG\tSCRIPT"]
    for split, members in (("train", range(1, 2 * n_pairs + 1)),):
        text, scp = [], []
        for i in members:
            conversation = f"ZH-CN_U{i:04d}"
            for u in range(2):
                utt = f"{conversation}_S0_{u}"
                text.append(f"{utt} 我 like 这个 project")
                scp.append(f"{utt} {utt}.wav")
                rows.append(f"{i:04d}\tF\t24\tFujian\t"
                            f"{'Android' if i % 2 else 'iphone'}\t"
                            f"{['work', 'study'][u]}\t{utt}.wav\t<MIX>\t我 like")
        (index_root / split).mkdir(parents=True)
        (index_root / split / "text").write_text("\n".join(text), encoding="utf-8")
        (index_root / split / "wav.scp").write_text("\n".join(scp), encoding="utf-8")
    info = tmp_path / "Information_Index.txt"
    info.write_text("\n".join(rows), encoding="utf-8")
    return {"data": {"index_root": str(index_root), "audio_root": str(tmp_path / "wav"),
                     "information_index": str(info), "splits": ["train"],
                     "sample_rate": 16000}}


def test_build_manifest_adds_dialogue_id_and_keeps_the_old_identities(tmp_path):
    cfg = _corpus(tmp_path)
    frame = build_manifest(cfg, hash_audio=False, probe_audio=False)

    assert {"dialogue_id", "corpus_speaker_id"} <= set(frame.columns)
    assert frame["dialogue_id"].notna().all()
    assert frame["conversation_id"].nunique() == 6
    assert frame["dialogue_id"].nunique() == 3          # two sides per dialogue

    # the pre-existing identities are exactly what parse_utterance_id says
    for _, row in frame.iterrows():
        conversation, speaker = parse_utterance_id(row["utterance_id"])
        assert row["conversation_id"] == conversation
        assert row["speaker_id"] == speaker

    # and a dialogue is two conversations, not one and not four
    per_dialogue = frame.groupby("dialogue_id")["conversation_id"].nunique()
    assert set(per_dialogue) == {2}


def test_dialogue_id_is_stable_across_rebuilds(tmp_path):
    cfg = _corpus(tmp_path)
    first = build_manifest(cfg, hash_audio=False, probe_audio=False)
    second = build_manifest(cfg, hash_audio=False, probe_audio=False)
    pd.testing.assert_series_equal(first["dialogue_id"], second["dialogue_id"])
    pd.testing.assert_series_equal(first["conversation_id"], second["conversation_id"])
    pd.testing.assert_series_equal(first["speaker_id"], second["speaker_id"])


def test_manifest_schema_extends_rather_than_replaces():
    for column in ("utterance_id", "conversation_id", "speaker_id"):
        assert column in MANIFEST_COLUMNS
    assert MANIFEST_COLUMNS.index("dialogue_id") > MANIFEST_COLUMNS.index("speaker_id")


def test_build_manifest_refuses_an_index_with_a_half_empty_dialogue(tmp_path):
    """Drop one participant: the manifest must fail, not pair the survivor."""
    cfg = _corpus(tmp_path, n_pairs=2)
    info = pd.read_csv(cfg["data"]["information_index"], sep="\t", dtype=str)
    kept = info[~info["WAVE ID"].str.startswith("ZH-CN_U0004")]
    kept.to_csv(cfg["data"]["information_index"], sep="\t", index=False)
    index = cfg["data"]["index_root"] + "/train"
    for name in ("text", "wav.scp"):
        lines = [ln for ln in open(f"{index}/{name}", encoding="utf-8").read().splitlines()
                 if not ln.startswith("ZH-CN_U0004")]
        open(f"{index}/{name}", "w", encoding="utf-8").write("\n".join(lines))
    with pytest.raises(DialoguePairingError, match="exactly two participants"):
        build_manifest(cfg, hash_audio=False, probe_audio=False)


# --------------------------------------------------------------------------
# the new artifact stage


def test_attach_dialogue_ids_changes_nothing_but_the_new_columns(tmp_path):
    from csasr.experiments.dialogue_manifest import attach_dialogue_ids

    cfg = _corpus(tmp_path)
    source = build_manifest(cfg, hash_audio=False, probe_audio=False)
    before = source.drop(columns=["dialogue_id", "corpus_speaker_id"])
    info = pd.DataFrame({
        "utterance_id": source["utterance_id"],
        "corpus_speaker_id": source["corpus_speaker_id"].map(lambda v: f"{int(v):04d}"),
    })
    after = attach_dialogue_ids(before, info)
    pd.testing.assert_frame_equal(after[before.columns], before)
    pd.testing.assert_series_equal(after["dialogue_id"], source["dialogue_id"])


def test_output_guard_refuses_protected_trees_and_overwrites(tmp_path):
    from csasr.experiments.dialogue_manifest import resolve_output
    from csasr.lss.manifest import manifest_path

    source = tmp_path / "cs_dialogue.parquet"
    source.write_bytes(b"")
    with pytest.raises(SystemExit, match="protected tree"):
        resolve_output("/mnt/data/x/artifacts_lss/alignments/new.parquet", source)
    with pytest.raises(SystemExit, match="immutable"):
        resolve_output(source, source)
    existing = tmp_path / "already.parquet"
    existing.write_bytes(b"")
    with pytest.raises(SystemExit, match="existing artifact"):
        resolve_output(existing, source)
    sidecar_only = tmp_path / "sidecar-only.parquet"
    manifest_path(sidecar_only).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="sidecar"):
        resolve_output(sidecar_only, source)
    assert resolve_output(tmp_path / "fresh.parquet", source) == tmp_path / "fresh.parquet"


def test_dialogue_derivation_fingerprint_binds_both_input_hashes():
    from csasr.experiments.dialogue_manifest import dialogue_derivation_fingerprint

    first, payload = dialogue_derivation_fingerprint(
        source_manifest_sha256="source-a", information_index_sha256="index-a")
    same, _ = dialogue_derivation_fingerprint(
        source_manifest_sha256="source-a", information_index_sha256="index-a")
    source_changed, _ = dialogue_derivation_fingerprint(
        source_manifest_sha256="source-b", information_index_sha256="index-a")
    index_changed, _ = dialogue_derivation_fingerprint(
        source_manifest_sha256="source-a", information_index_sha256="index-b")
    assert first == same
    assert first not in {source_changed, index_changed}
    assert payload["information_index_sha256"] == "index-a"
    assert payload["source_manifest_sha256"] == "source-a"


def test_dialogue_manifest_records_information_index_as_hashed_parent(
        tmp_path, monkeypatch, capsys):
    from csasr.experiments import dialogue_manifest as command
    from csasr.lss import manifest as artifact_manifest
    from csasr.utils.hashing import sha256_file

    cfg = _corpus(tmp_path, n_pairs=3)
    built = build_manifest(cfg, hash_audio=False, probe_audio=False)
    source = tmp_path / "source.parquet"
    built.drop(columns=["dialogue_id", "corpus_speaker_id"]).to_parquet(
        source, index=False)
    monkeypatch.setattr(command, "load_config", lambda _: cfg)
    output = tmp_path / "dialogue-v2"
    assert command.main([
        "--config", "ignored.yaml", "--source-manifest", str(source),
        "--information-index", cfg["data"]["information_index"],
        "--output-dir", str(output),
    ]) == 0
    capsys.readouterr()

    target = output / command.MANIFEST_NAME
    sidecar = artifact_manifest.load(target)
    assert sidecar is not None
    parents = {p["path"]: p["sha256"] for p in sidecar["parent_artifacts"]}
    assert parents[str(source)] == sha256_file(source)
    assert parents[cfg["data"]["information_index"]] == \
        sha256_file(cfg["data"]["information_index"])
    assert len(sidecar["dialogue_derivation_fingerprint"]) == 64
