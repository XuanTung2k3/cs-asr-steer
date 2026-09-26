"""Attach immutable repair provenance to the 76 newly decoded result records."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("results/basis_a3_raw_cond_scope_depth")
MASK_HASH = "sha256:10110ab016d7895664d78240717c88f74011353a59eadd3dde73910bb3d2636b"
PROTOCOL_COMMIT = "69a5aa0933c6f8be43752d6a28a414a92c5f875e"


def main() -> None:
    paths = []
    for dataset, r1_job, r2_job in (("seame_dev_man", "51682", "51682"), ("seame_dev_sge", "51683", "51683")):
        paths.extend(sorted((ROOT / "raw_r1" / dataset).glob("L*/Raw_encoder_L*_oracle_local_rho0.5.json")))
        paths.extend(sorted((ROOT / "raw_r2" / dataset).glob("L*/Raw_encoder_L*_oracle_local_rho*.json")))
    paths = [
        p for p in paths
        if json.loads(p.read_text()).get("scope") == "oracle_local"
        and float(json.loads(p.read_text())["rho"]) in (0.25, 0.5, 1.0)
        and p.parent.parent.name in ("seame_dev_man", "seame_dev_sge")
    ]
    if len(paths) != 76:
        raise RuntimeError(f"expected 76 repair records, found {len(paths)}")
    for path in paths:
        data = json.loads(path.read_text())
        data["repair_provenance"] = {
            "status": "canonical_repair",
            "repair_protocol": "results/basis_a3_raw_cond_scope_depth/repair/REPAIR_PROTOCOL.json",
            "repair_protocol_commit": PROTOCOL_COMMIT,
            "mask_definition": "target-segment union alignment",
            "mask_implementation_hash": MASK_HASH,
            "boundary_tolerance_ms": 80,
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"annotated {len(paths)} repaired records")


if __name__ == "__main__":
    main()
