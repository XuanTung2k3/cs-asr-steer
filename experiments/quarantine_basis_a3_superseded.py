"""Move exactly the 76 pre-repair SEAME encoder-local result files to quarantine."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("results/basis_a3_raw_cond_scope_depth")
DEST = ROOT / "quarantine/superseded_noncanonical_mask"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main() -> None:
    sources = []
    for dataset, r1_job, r2_job in (("seame_dev_man", "51649", "51654"), ("seame_dev_sge", "51652", "51655")):
        for path in sorted((ROOT / "raw_r1" / dataset).glob("L*/Raw_encoder_L*_oracle_local_rho0.5.json")):
            sources.append((path, r1_job, "r1"))
        for path in sorted((ROOT / "raw_r2" / dataset).glob("L*/Raw_encoder_L*_oracle_local_rho*.json")):
            payload = json.loads(path.read_text())
            if float(payload["rho"]) in (0.25, 1.0) and int(payload["layer"]) in (2, 16, 21):
                sources.append((path, r2_job, "r2"))
    if len(sources) != 76:
        raise RuntimeError(f"expected exactly 76 superseded files, found {len(sources)}")
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for source, job, stage in sources:
        relative = source.relative_to(ROOT)
        destination = DEST / relative
        if destination.exists():
            raise RuntimeError(f"destination already exists: {destination}")
        payload = json.loads(source.read_text())
        old_hash = digest(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        records.append({
            "original_path": str(relative),
            "quarantine_path": str(destination.relative_to(ROOT)),
            "sha256": old_hash,
            "dataset": payload["dataset"],
            "side": payload["metrics"]["side"],
            "direction": payload["direction"],
            "layer": payload["layer"],
            "scope": payload["scope"],
            "rho": payload["rho"],
            "source_job": job,
            "source_stage": stage,
            "source_commit": "d4012bd5bb085ac983c2acb1eeed1bbf400ad17e" if job in ("51649", "51652") else "aa71f217f5103844bca713664a8e98a9ae8c384e",
            "mask_semantics": "second-boundary alignment",
        })
    manifest = {
        "schema_version": "basis_a3_superseded_noncanonical_mask_v1",
        "reason": "SEAME encoder Oracle-local outputs predate target-segment-union harmonization",
        "count": len(records),
        "records": records,
    }
    (DEST / "superseded_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"quarantined {len(records)} files")


if __name__ == "__main__":
    main()
