# BASIS-A4 Interrupted Jobs

| Job | Evidence | Action | Output disposition |
|---|---|---|---|
| 52083 | `sacct`: CANCELLED+, elapsed 00:05:46; no measurable completed-cell progress after the early invalid-layout preparation | canceled cleanly | early outputs retained outside the accepted namespace and quarantined |
| 52085 | `sacct`: CANCELLED+, elapsed 00:16:35; only a small partial CS Conditioning band was written | canceled cleanly after the A4 gate failure | partial CS outputs are not accepted; clean rerun is required |

The completed MAN/SGE cells from the same pre-freeze execution are also not
automatically accepted because they were generated before the clean commit.
They remain available for audit but are superseded by clean reruns.

The 24 early invalid-layout files are retained under
`results/basis_a4/quarantine/whisper_conditioning_legacy_layout/`.
