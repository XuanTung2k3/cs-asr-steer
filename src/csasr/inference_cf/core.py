"""Pure P0 contracts shared by preparation, execution, and CPU audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

VERSION = "p0_cross_k1_v1"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return "sha256:" + hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def condition_tokens(processor):
    from csasr.data.alignment import build_prefix
    return {"c0": build_prefix(processor, "zh"),
            "cM": build_prefix(processor, "zh"),
            "cE": build_prefix(processor, "en")}


def conditions_identical(conditions, a="c0", b="cM"):
    return tuple(conditions[a]) == tuple(conditions[b])


def aligned_input(prompt, content):
    if not prompt:
        raise ValueError("empty prompt")
    if any(not isinstance(x, int) for x in prompt + content):
        raise ValueError("non-integer token")
    return prompt + content


def support(scores, y_e, y_m):
    if not all(math.isfinite(float(scores[x])) for x in ("sE_yE", "sE_yM", "sM_yE", "sM_yM")):
        raise ValueError("nonfinite score")
    return {"collision": y_e == y_m,
            "q_own": scores["sE_yE"] - scores["sM_yM"],
            "q_cross": None if y_e == y_m else .5 * ((scores["sE_yE"] - scores["sE_yM"])
                                                     + (scores["sM_yM"] - scores["sM_yE"]))}


def permutation(ids):
    ids = list(ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("need distinct rows")
    return {x: ids[(i + 1) % len(ids)] for i, x in enumerate(ids)}


def cache_key(*, model_revision, audio_sha256, condition, prompt_tokens, content_prefix,
              logical_position, layer, site, beam_lineage, scorer_version, decode_config):
    return digest(locals())


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical(obj) + "\n")
    tmp.replace(path)


def validated_row(path, identity, manifest_hash):
    p = Path(path)
    if not p.exists():
        return None
    row = json.loads(p.read_text())
    if row.get("identity") != identity or row.get("manifest_hash") != manifest_hash:
        raise ValueError(f"stale or mismatched row {p}")
    return row
