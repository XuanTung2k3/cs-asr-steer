#!/usr/bin/env python
"""P2-1 post-training verification (GPU, no training): every saved epoch adapter reloads with its recorded hash and
re-decodes the first N dev150 utterances (frozen LID, beam-5, canonical environment) token for token as recorded during
training; backbone weights equal the P0 anchor before and after. Writes p2_1/verify.json (write-once)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, cfg, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import cublas_consistent, git_state, loaded_cuda_libs, now, sha, slurm, weights_hash  # noqa: E402

N_ROWS = 20


def main():
    import torch
    from reft.adapters import make_adapter
    from reft.auto_policy import decode_frozen_lid
    from reft.train_supervised import c21, check_freeze, load_backbone
    C1 = c21()
    fz = check_freeze()
    b, env = load_backbone()
    w0 = weights_hash(b.model)
    dm = read_json(art("p2_0/dev_manifest.json"))["rows"][:N_ROWS]
    inv = {v: k for k, v in b.model.generation_config.lang_to_id.items()}
    res = {}
    for m in ("R2", "L1"):
        R = art(f"p2_1/{m}")
        man = read_json(R / "manifest.json")
        res[m] = {}
        for h in man["history"][1:]:
            ep = h["epoch"]
            p = R / f"epoch{ep}" / "adapter.pt"
            ad = make_adapter(m, b.model, C1, b.device)
            ad.load_state_dict(torch.load(p))
            rec = {x["id"]: x for x in read_jsonl(R / f"epoch{ep}" / "dev_beam5.jsonl")}
            eq = 0
            for r in dm:
                lid = {"token": r["frozen_lid_token"], "language": inv[r["frozen_lid_token"]]}
                d, _ = decode_frozen_lid(b, r["audio_path"], lid, 5, ad)
                eq += d["tokens"] == rec[r["utterance_id"]]["tokens"]
            res[m][ep] = {"adapter_sha_matches_manifest": sha(p) == h["adapter_sha256"], "reload_decode_equal": [eq, len(dm)],
                          "n_trainable": ad.n_trainable()}
    w1 = weights_hash(b.model)
    libs = loaded_cuda_libs()
    ok = (w0 == w1 == cfg()["model"]["expected_weights_hash"] and cublas_consistent(libs)
          and all(v["adapter_sha_matches_manifest"] and v["reload_decode_equal"][0] == len(dm) for x in res.values() for v in x.values()))
    out = {"schema": "reft_p2_1_verify_v1", "utc": now(), **fz, "git": git_state(), "slurm": slurm(), "results": res,
           "weights_before": w0, "weights_after": w1, "cuda_libs": libs, "all_ok": ok}
    write_json(art("p2_1/verify.json"), out)
    print(json.dumps({"all_ok": ok, "results": res}, indent=1))


if __name__ == "__main__":
    main()
