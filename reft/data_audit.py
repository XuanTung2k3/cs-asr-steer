#!/usr/bin/env python
"""S0 data audit (CPU).

  p0  -- verify the ACL-0 dev150 panel (identity, role, disjointness, audio hashes, sealed ACL-0 outputs) and write the
         label-free P0 decode panel. Reads only ID/audio columns of D-dev-select; no transcript is read.
  p1  -- build the P1 basis-training panel from the training roles (loc-train U util-train), its dialogue-disjoint
         construction/audit split, and the label-derived targets (training labels only).

Locked roles (D-test, D-dev-confirm, router-calib) are never opened; their dialogues are known from assignment.json only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, cfg, fillers, read_json, write_json  # noqa: E402
from reft.provenance import environment, git_state, now, sha  # noqa: E402

ID_COLS = ["utterance_id", "conversation_id", "speaker_id", "dialogue_id", "corpus_speaker_id", "duration_sec",
           "audio_sha256", "audio_path", "role"]
KEYS = ["dialogue_id", "conversation_id", "speaker_id", "corpus_speaker_id", "audio_sha256", "utterance_id"]


def role_rows(role: str, cols=ID_COLS):
    import pyarrow.parquet as pq
    C = cfg()
    if role in C["data"]["locked_roles_never_opened"]:
        raise RuntimeError(f"locked role {role} must not be opened")
    return pq.read_table(Path(C["data"]["role_dir"]) / f"role_{role}.parquet", columns=cols).to_pylist()


def assignment() -> dict:
    return read_json(Path(cfg()["data"]["role_dir"]) / "assignment.json")["assignment"]


def cross(a, b) -> dict:
    return {k: len({r[k] for r in a} & {r[k] for r in b}) for k in KEYS}


def acl0_repo_root() -> Path:
    return Path(cfg()["data"]["acl0_dir"]).parents[2]


# ====================================================================== P0

def audit_p0():
    C = cfg()
    A = Path(C["data"]["acl0_dir"])
    checks = {}
    checks["acl0_panels_sha_ok"] = sha(A / "panels.json") == C["data"]["acl0_panels_sha256"]
    checks["acl0_output_seal_sha_ok"] = sha(A / "output_seal.json") == C["data"]["acl0_output_seal_sha256"]
    seal = read_json(A / "output_seal.json")
    sel = read_json(A / "selection.json")
    ep = {a: sel["arms"][a]["selected"]["epoch"] for a in ("H1", "A1")}
    need = {"B0": "results/research_review/acl0/e0/dev_beam5.json.gz",
            "H1": f"results/research_review/acl0/H1/epoch{ep['H1']}/dev.json.gz",
            "A1": f"results/research_review/acl0/A1/epoch{ep['A1']}/dev.json.gz"}
    rr = acl0_repo_root()
    checks["acl0_sealed_outputs_ok"] = {k: seal["files"].get(p) == sha(rr / p) for k, p in need.items()}
    e0m = read_json(A / "e0/manifest.json")
    checks["acl0_b0_prompt_forced_zh"] = e0m["env"]["prompt_ids"]
    P = read_json(A / "panels.json")
    dev = P["dev"]
    ids = [r["utterance_id"] for r in dev]
    role = {r["utterance_id"]: r for r in role_rows(C["data"]["dev_role"])}
    checks["n_panel"] = len(dev)
    checks["panel_ids_unique"] = len(set(ids)) == len(ids)
    checks["all_in_dev_role"] = all(i in role for i in ids)
    fields = ["dialogue_id", "conversation_id", "speaker_id", "audio_sha256", "audio_path", "duration_sec"]
    checks["panel_fields_match_role_manifest"] = sum(all(role[r["utterance_id"]][f] == r[f] for f in fields if f in r)
                                                     for r in dev)
    asg = assignment()
    dlg = sorted({r["dialogue_id"] for r in dev})
    checks["panel_dialogues"] = len(dlg)
    checks["panel_dialogue_roles"] = sorted({asg[d] for d in dlg})
    train = []
    for r in C["data"]["train_roles"]:
        train += role_rows(r)
    prow = [role[i] for i in ids]
    checks["panel_vs_train_roles_shared"] = cross(prow, train)
    checks["dev_role_vs_train_roles_shared"] = cross(list(role.values()), train)
    allsha = collections.Counter(r["audio_sha256"] for r in list(role.values()) + train)
    checks["panel_audio_sha_duplicates_within_panel"] = len(ids) - len({r["audio_sha256"] for r in prow})
    checks["panel_audio_sha_seen_more_than_once_in_dev_or_train"] = sum(allsha[r["audio_sha256"]] > 1 for r in prow)
    # speaker (corpus id) linkage across dialogues within the dev role
    spk = collections.defaultdict(set)
    for r in role.values():
        spk[r["corpus_speaker_id"]].add(r["dialogue_id"])
    checks["dev_corpus_speakers_in_multiple_dialogues"] = sum(len(v) > 1 for v in spk.values())
    locked = {d for d, ro in asg.items() if ro in C["data"]["locked_roles_never_opened"]}
    checks["panel_dialogues_in_locked_roles"] = len(set(dlg) & locked)
    strata = collections.Counter(read_json(A / "panel_strata.json")["strata"][i] for i in ids)
    ok = (checks["acl0_panels_sha_ok"] and checks["acl0_output_seal_sha_ok"] and all(checks["acl0_sealed_outputs_ok"].values())
          and checks["panel_ids_unique"] and checks["all_in_dev_role"] and checks["panel_fields_match_role_manifest"] == len(dev)
          and checks["panel_dialogue_roles"] == [C["data"]["dev_role"]] and all(v == 0 for v in checks["panel_vs_train_roles_shared"].values())
          and checks["panel_audio_sha_duplicates_within_panel"] == 0 and checks["panel_dialogues_in_locked_roles"] == 0)
    rec = {"schema": "reft_p0_data_audit_v1", "utc": now(), "git": git_state(), "valid": bool(ok), "checks": checks,
           "acl0_selected_epochs": ep, "acl0_manifest_strata_flags": dict(strata),
           "exposure": C["data"]["dev_exposure"],
           "overlapping_segments": "not checkable: the role manifest has no segment start/end times; exact audio duplicates "
                                   "are checked by sha256 and recording identity by conversation/speaker IDs",
           "cached_auto_transcripts": "none: ACL-0 E0 decoded with the forced prompt "
                                      f"{e0m['env']['prompt_ids']}; no Auto beam-5 decode of this panel exists (ACL-BASE-0 not run)",
           "reference_columns_read": False}
    write_json(art("p0/data_audit.json"), rec)
    write_json(art("p0/panel.json"), {"schema": "reft_p0_panel_v1", "source": "ACL-0 panels.json dev (A3-ACL0-DEV150)",
                                      "rows": [{k: r[k] for k in ("utterance_id", "dialogue_id", "conversation_id", "speaker_id",
                                                                  "duration_sec", "audio_path", "audio_sha256")} for r in dev]})
    print({"valid": ok, **{k: v for k, v in checks.items()}})


# ====================================================================== P1

def round_robin(rows, quota, seed):
    import numpy as np
    out, short = [], {}
    rng = np.random.default_rng(seed)
    for s, q in quota.items():
        by = collections.defaultdict(list)
        for r in sorted((r for r in rows if r["stratum"] == s), key=lambda r: r["utterance_id"]):
            by[r["dialogue_id"]].append(r)
        order = {d: [by[d][i] for i in rng.permutation(len(by[d]))] for d in sorted(by)}
        pos, sel = {d: 0 for d in order}, []
        while len(sel) < q and any(pos[d] < len(order[d]) for d in order):
            for d in order:
                if len(sel) < q and pos[d] < len(order[d]):
                    sel.append(order[d][pos[d]])
                    pos[d] += 1
        short[s] = q - len(sel)
        out += sel
    return out, short


def split_dialogues(rows, frac):
    dl = sorted({r["dialogue_id"] for r in rows}, key=lambda d: hashlib.sha256(f"reft-p1-split:{d}".encode()).hexdigest())
    n = collections.Counter(r["dialogue_id"] for r in rows)
    audit, tot = [], 0
    for d in dl:
        if tot >= frac * len(rows):
            break
        audit.append(d)
        tot += n[d]
    return set(audit)


def audit_p1():
    from transformers import WhisperTokenizer
    from reft.text import target_text, units, utterance_stratum
    C = cfg()
    D = C["data"]
    PP = D["p1_panel"]
    FILL = fillers()
    train = []
    for r in D["train_roles"]:
        train += role_rows(r, ID_COLS + ["transcript_raw"])
    dev = role_rows(D["dev_role"])
    acl0_train = {r["utterance_id"] for r in read_json(Path(D["acl0_dir"]) / "panels.json")["train"]}
    tok = WhisperTokenizer.from_pretrained(C["model"]["path"], local_files_only=True)
    lo, hi = D["duration_sec"]
    elig, excl = [], collections.Counter()
    for r in train:
        if r["utterance_id"] in acl0_train:
            excl["acl0_train_panel"] += 1
            continue
        if not (lo <= r["duration_sec"] <= hi):
            excl["duration"] += 1
            continue
        t = target_text(r["transcript_raw"])
        if not t or units(t) != units(r["transcript_raw"]):
            excl["empty_or_unit_mismatch"] += 1
            continue
        if len(tok.encode(t, add_special_tokens=False)) > D["max_target_tokens"]:
            excl["too_many_tokens"] += 1
            continue
        st = utterance_stratum(r["transcript_raw"], FILL)
        if st["stratum"] not in PP["quota"] or st["n_sub_en"] + st["n_zh"] < PP["min_units"]:
            excl["stratum_or_min_units"] += 1
            continue
        elig.append({**r, **st})
    sel, short = round_robin(elig, PP["quota"], C["seed"])
    audit_d = split_dialogues(sel, PP["audit_fraction"])
    asg = assignment()
    locked = {d for d, ro in asg.items() if ro in D["locked_roles_never_opened"]}
    p0 = read_json(art("p0/panel.json"))["rows"]
    dlg = {r["dialogue_id"] for r in sel}
    con = [r for r in sel if r["dialogue_id"] not in audit_d]
    aud = [r for r in sel if r["dialogue_id"] in audit_d]
    checks = {"panel_vs_dev_role_shared": cross(sel, dev), "panel_vs_p0_panel_shared": cross(sel, p0),
              "panel_dialogue_roles": sorted({asg[d] for d in dlg}), "panel_dialogues_in_locked_roles": len(dlg & locked),
              "construction_vs_audit_shared": cross(con, aud),
              "audio_sha_duplicates": len(sel) - len({r["audio_sha256"] for r in sel}),
              "panel_vs_acl0_train_shared_ids": len({r["utterance_id"] for r in sel} & acl0_train)}
    ok = (all(v == 0 for v in checks["panel_vs_dev_role_shared"].values()) and all(v == 0 for v in checks["panel_vs_p0_panel_shared"].values())
          and set(checks["panel_dialogue_roles"]) <= set(D["train_roles"]) and checks["panel_dialogues_in_locked_roles"] == 0
          and checks["construction_vs_audit_shared"]["dialogue_id"] == 0 and checks["audio_sha_duplicates"] == 0
          and checks["panel_vs_acl0_train_shared_ids"] == 0)

    def comp(rows):
        return {"n": len(rows), "dialogues": len({r["dialogue_id"] for r in rows}),
                "strata": dict(collections.Counter(r["stratum"] for r in rows)),
                "hours": round(sum(r["duration_sec"] for r in rows) / 3600, 3),
                "substantive_en_units": sum(r["n_sub_en"] for r in rows), "zh_units": sum(r["n_zh"] for r in rows)}
    keep = ["utterance_id", "dialogue_id", "conversation_id", "speaker_id", "duration_sec", "audio_path", "audio_sha256", "stratum"]
    panel = [{**{k: r[k] for k in keep}, "partition": "audit" if r["dialogue_id"] in audit_d else "construction"} for r in sel]
    targets = {}
    for r in sel:
        t = target_text(r["transcript_raw"])
        targets[r["utterance_id"]] = {"target_text": t, "target_ids": tok.encode(t, add_special_tokens=False) + [tok.eos_token_id],
                                      "reference_raw": r["transcript_raw"]}
    rec = {"schema": "reft_p1_data_audit_v1", "utc": now(), "git": git_state(), "env": environment(), "valid": bool(ok),
           "checks": checks, "train_pool": len(train), "eligible": len(elig), "eligible_strata": dict(collections.Counter(r["stratum"] for r in elig)),
           "exclusions": dict(excl), "quota_shortfall": short, "panel": comp(sel), "construction": comp(con), "audit": comp(aud),
           "audit_dialogues": len(audit_d), "seed": C["seed"],
           "label_use": "training references only (loc-train U util-train); dev and locked roles: ID columns / assignment only"}
    write_json(art("p1/data_audit.json"), rec)
    write_json(art("p1/panel.json"), {"schema": "reft_p1_panel_v1", "rows": panel})
    write_json(art("p1/targets.json"), {"schema": "reft_p1_targets_v1", "note": "training labels; read only after decoding",
                                        "targets": targets})
    print({k: rec[k] for k in ("valid", "checks", "eligible", "eligible_strata", "exclusions", "quota_shortfall", "panel",
                               "construction", "audit", "audit_dialogues")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["p0", "p1"])
    a = ap.parse_args(argv)
    {"p0": audit_p0, "p1": audit_p1}[a.cmd]()


if __name__ == "__main__":
    main()
