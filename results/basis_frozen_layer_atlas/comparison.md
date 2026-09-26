# BASIS-A2 — Frozen all-layer steering response atlas

All 32 decoder layers, five frozen direction families, and four doses were evaluated on the ten-utterance D-dev-select micro-panel. The complete table is in `main_performance_table.md`/`.csv`; machine-readable summaries are in `summary.json`, `free_decode_summary.json`, `teacher_forced_summary.json`, `projection_stats.json`, and `geometry/`.

At rho=0.5 averaged over layers, Raw/Local produced 29/6 and 26/7 corrections/corruptions, while Conditioning produced 170/65 and the largest PIER gain. At higher rho, damage grew faster than correction for every family. Conditioning had its strongest exploratory panel signal at L26–L27; this is not a validated layer selection.

Raw and Local were near-identical in the panel. Residualization removed at most 2.17% of normalized Raw energy across layers and reduced the Raw/Cond basis condition number to 1 for the Local/Cond basis. The rank-2 spaces were numerically identical (maximum principal angle 3.08e-6 degrees; projection distance 1.08e-15), so residualization changes coordinate conditioning rather than available representational information. The fixed coefficients still produced a small RC–LC physical angle that varied by layer.

PCA is deferred because no larger compatible cached representation sample was available and no extra GPU sweep was authorized. Projection distributions and teacher-forced token/representation diagnostics are descriptive; causal claims use free-decoding outputs only. Overall WER is n/a because no frozen canonical overall WER implementation exists.
