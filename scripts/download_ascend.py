#!/usr/bin/env python3
"""Reproducible download + preparation of the official ASCEND corpus (CAiRE/ASCEND).

Companion to `docs/current/BASIS_A6_ASCEND_DATA_SPEC.md`. This script ONLY downloads and
snapshots the corpus and records provenance. It performs **no** eligibility filtering,
**no** construction/eval subset selection, and **no** model inference. Subset freezing is a
separate, later step (see the data spec §6–§7). It fails loudly if the upstream structure is
incompatible with the frozen protocol rather than silently adapting.

Usage:
    python scripts/download_ascend.py                # default paths under data/external/ASCEND
    python scripts/download_ascend.py --out data/external/ASCEND --revision <sha>

Notes:
- Do NOT commit dataset audio to git. Only the small JSON provenance record
  (`ASCEND_DOWNLOAD_PROVENANCE.json`) is intended to be tracked; the `dataset/` and
  `hf_cache/` trees must be git-ignored.
- ASCEND `test` is DO-NOT-READ / DO-NOT-USE for BASIS-A6. This script snapshots the whole
  DatasetDict (so fingerprints are complete and reproducible) but writes a guard marker that
  downstream code must honour; nothing here reads test rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASET_REPO = "CAiRE/ASCEND"
EXPECTED_SPLITS = ("train", "validation", "test")
# Approximate upstream sizes (protocol reference, BASIS_A6_ASCEND_DATA_SPEC §4). Used only
# for a loud sanity warning, never to silently reshape the data.
EXPECTED_SIZES = {"train": 9869, "validation": 1130, "test": 1315}
DO_NOT_USE_SPLITS = ("test",)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _code_hash() -> str:
    return "sha256:" + _sha256_file(Path(__file__))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/external/ASCEND",
                    help="output root (default: data/external/ASCEND)")
    ap.add_argument("--revision", default=None,
                    help="optional pinned dataset revision (commit sha / tag)")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="continue even if split sizes differ from the protocol reference")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"[FATAL] `datasets` not importable: {exc}", file=sys.stderr)
        return 2

    from datasets import load_dataset

    out = Path(args.out)
    cache_dir = out / "hf_cache"
    save_dir = out / "dataset"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading {DATASET_REPO} (revision={args.revision or 'default'})")
    ds = load_dataset(
        DATASET_REPO,
        cache_dir=str(cache_dir),
        revision=args.revision,
    )

    got_splits = tuple(ds.keys())
    missing = [s for s in EXPECTED_SPLITS if s not in got_splits]
    extra = [s for s in got_splits if s not in EXPECTED_SPLITS]
    if missing:
        print(f"[FATAL] upstream is missing required splits {missing}; "
              f"got {got_splits}. Refusing to proceed (protocol incompatible).",
              file=sys.stderr)
        return 3
    if extra:
        # Not fatal, but must be recorded and reviewed by a human.
        print(f"[WARN] upstream has unexpected extra splits {extra}; recorded in provenance.")

    sizes = {s: len(ds[s]) for s in got_splits}
    print("[info] split sizes:", sizes)

    size_ok = True
    for s, n in EXPECTED_SIZES.items():
        if s in sizes and sizes[s] != n:
            size_ok = False
            print(f"[WARN] split '{s}' size {sizes[s]} != protocol reference {n}")
    if not size_ok and not args.allow_size_mismatch:
        print("[FATAL] split sizes differ from the frozen protocol reference. "
              "Re-run with --allow-size-mismatch ONLY after a human updates the data spec "
              "with the new upstream revision + sizes.", file=sys.stderr)
        return 4

    print(f"[info] saving DatasetDict to {save_dir}")
    ds.save_to_disk(str(save_dir))

    # Fingerprints: HF exposes a per-split content fingerprint; also capture feature schema.
    fingerprints = {}
    features = {}
    for s in got_splits:
        fingerprints[s] = getattr(ds[s], "_fingerprint", None)
        features[s] = {name: str(feat) for name, feat in ds[s].features.items()}

    provenance = {
        "schema_version": "ascend_download_provenance_v1",
        "dataset_repo": DATASET_REPO,
        "requested_revision": args.revision,
        "expected_splits": list(EXPECTED_SPLITS),
        "observed_splits": list(got_splits),
        "expected_sizes_reference": EXPECTED_SIZES,
        "observed_sizes": sizes,
        "size_matches_reference": size_ok,
        "hf_fingerprints": fingerprints,
        "feature_schema": features,
        "do_not_use_splits": list(DO_NOT_USE_SPLITS),
        "save_dir": str(save_dir),
        "cache_dir": str(cache_dir),
        "preparation_code_hash": _code_hash(),
    }
    prov_path = out / "ASCEND_DOWNLOAD_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f"[info] wrote provenance -> {prov_path}")

    # Guard marker for downstream code: test must never be read for BASIS-A6.
    guard = out / "DO_NOT_USE_TEST.marker"
    guard.write_text(
        "ASCEND 'test' split is DO-NOT-READ / DO-NOT-USE for BASIS-A6.\n"
        "Construction: TRAIN only. Evaluation panel: VALIDATION only.\n"
    )
    print(f"[info] wrote guard -> {guard}")
    print("[done] ASCEND download + snapshot complete. Subset freezing is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
