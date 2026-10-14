# CS-ASR-ReFT status

| Stage | State | Evidence (local artifacts under `/mnt/data/tungnx/cs-asr-reft/`) |
|---|---|---|
| S0 | done | branch `cs-asr-reft` from `main` `db365d8f`; scope and protocol frozen (`freeze/p0.json`) |
| P0 | **`P0_PASS_RELEVANT_OPPORTUNITIES`** (unchanged by AMENDMENT L31) | `p0/evaluation.json`. Attempt 1 was BLOCKED by a hash-convention defect and is kept as a superseded record (amendment r1). |
| P1 @ layer 24 (original) | **historical, incomplete**: capture (59135) and basis (59136) built; held-out evaluation never completed (59137 crashed before writing) | `p1/` (untouched; no feasibility outcome exists) |
| P1 @ layer 31 (AMENDMENT L31, active) | **`P1_SUBSPACE_WEAK_OR_NARROW`**: all gates pass except preservation (worst-of-8 flip 0.136 > 0.10) | `p1/l31/evaluation.json`; basis sha256 `f020e93b…`; see `P0P1_RESULT.md` |
| P2 | not started; not authorized. The frozen verdict forbids automatic controller training. Recommended first: CPU selectivity audit (`P0P1_RESULT.md` §5) | — |
