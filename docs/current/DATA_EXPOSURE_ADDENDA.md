# DATA EXPOSURE — addenda

`DATA_EXPOSURE.md` stays the authoritative ledger up to the SRD2-G0 freeze. It is byte-pinned by the frozen SRD2-G0
contract (`configs/inference_cf/srd2_g0.json` → `source_sha256`), so it is not edited. Later exposures are
registered append-only in `DATA_EXPOSURE_ADDENDA.json`, which records the ledger's hash.

Any new D-dev-select sample selector must exclude `csasr.inference_cf.exposure_registry.exposed_ids()`. That set
combines two sources:
- the documented exposure registry inventoried at the SRD2-G0 freeze (300 IDs);
- every addendum entry.

The reader hash-checks the ledger, the registry and the addenda, and refuses to run if any of them changed.

| Addendum | Stage | Role | IDs | Exposure |
|---|---|---|---|---|
| A1-SRD2-G0-400 | SRD2-G0 (2026-10-09) | D-dev-select, 20 dialogues | 400 (`sha256:ff2e3054…`) | Single-query intervened pulses (Slurm 58300/58314). References opened post-seal for lexical evaluation, filtered to these IDs. Now exposed development data, not fresh validation |

The DIR-SPRINT0 feasibility and D5 provider sessions read no D-dev-select audio, reference or label. They add no
exposure entry.
