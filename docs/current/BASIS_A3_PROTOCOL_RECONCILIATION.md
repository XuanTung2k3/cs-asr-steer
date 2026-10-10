# BASIS-A3 Protocol Reconciliation

Status: **REPAIRED WITH DOCUMENTED PROVENANCE**. The historical mismatch and
the exact post-run harmonization are both retained below.

## Freeze and execution chronology

| Snapshot | Role | Commit | Config hash | Evidence |
|---|---|---|---|---|
| Original protocol freeze | Initial A3 freeze | `5a8b068` | recorded in the freeze artifacts | `docs/current/BASIS_A3_RAW_COND_SCOPE_DEPTH_SPEC.md` and freeze history |
| CS R1 | Raw encoder/decoder, CS-Dialogue | `5a5f22b` | `4b7e536c...` | accepted R1 manifests for jobs 51638/51639 |
| SEAME R1 | Raw encoder/decoder, dev-man/dev-sge | `d4012bd` | `570aa275...` | accepted R1 manifests for jobs 51649–51652 |
| R2 + Conditioning | all three datasets | `aa71f21` | `570aa275...` | accepted manifests for jobs 51653–51655 |

The protocol/spec was resealed after execution began. The original manifests
remain historical evidence and are not rewritten.

## Relevant changes

| Change | Commit | Category | Affects scientific outputs? | Evidence |
|---|---|---|---|---|
| Initial A3 protocol/spec freeze | `5a8b068` | A / protocol | Yes, defines the target | Freeze/spec history |
| Protocol hash/manifest reseal and execution bookkeeping | `df4b078`, `754c07e`, `5a5f22b` | B / operational | No by itself | Git history and CS manifests |
| SEAME encoder Oracle-local span construction changed from the second segment boundary to the union of target segments, with the frozen 80 ms silence/frame conversion | `0b8d1f3` (present in `d4012bd`) | **A / scientific-semantic** | **Yes** | `git diff 5a5f22b d4012bd -- experiments/basis_a3.py`; `_encoder_span` implementation |
| Encoder eligibility/alignment metadata and 1000-frame bookkeeping | `0b8d1f3` | B / operational | No, metadata only | `experiments/basis_a3.py` and accepted manifests |
| Parser/CSV field compatibility and labels | `d4012bd` | C / analysis-only | No | `experiments/analyze_basis_a3.py` diff |
| `--side both` and combined R2+Conditioning orchestration | `aa71f21` | B / operational | No | `experiments/basis_a3.py` diff; jobs 51653–51655 |
| R2 layer-selection freeze | `3b86926` | B / protocol bookkeeping | No after selection | `selection/r2_layers.json`, timestamped before R2 execution |

The first row in the semantic change is decisive. The old SEAME local mask
used the second boundary from the segment payload. The later implementation
uses all `target=true` segments, reconstructs their acoustic-time spans with
the payload's silence interval, and converts that union using the frozen
80 ms tolerance. This changes which encoder frames are intervened on for
SEAME Oracle-local outputs. CS uses its separate existing-CTC span path, and
Global masks do not use this span construction.

## Affected existing result blocks

No results are deleted, relabeled as equivalent, or rerun in this repair.
If the study is later harmonized to one scientific snapshot, the minimum
affected block set is:

- SEAME dev-man encoder, Raw, Oracle-local, R1, rho=.5: all 32 layers.
- SEAME dev-sge encoder, Raw, Oracle-local, R1, rho=.5: all 32 layers.
- SEAME dev-man encoder, Raw, Oracle-local, R2: L2/L16/L21 at rho=.25 and 1.0 (6 cells; rho=.5 is the R1 reuse).
- SEAME dev-sge encoder, Raw, Oracle-local, R2: L2/L16/L21 at rho=.25 and 1.0 (6 cells; rho=.5 is the R1 reuse).

Total: **76 accepted decoded blocks** are affected by this specific semantic
change. Decoder blocks, Conditioning blocks, CS encoder blocks, and SEAME
Global encoder blocks are not implicated by this `_encoder_span` difference.

## Post-run harmonization

The canonical repair definition is the later **target-segment union
alignment**: union every `target=true` segment after reconstructing acoustic
time with the frozen `silence_seconds=0.08`, then convert the union to encoder
frames with the unchanged 80 ms boundary convention. This agrees with the
current A3 spec, the implementation in `d4012bd`/`aa71f21`, the focused local
mask test, and the R2 execution snapshot.

The repair protocol was frozen and committed at `69a5aa0` in
`results/basis_a3_raw_cond_scope_depth/repair/REPAIR_PROTOCOL.json`. Acceptance
job `51680` passed all five real-model checks. The exact 76 affected cells were
rerun in two MIG jobs: `51682` for dev-man and `51683` for dev-sge. Both
completed successfully. R2 rho=.5 was not separately decoded; it is reused
from the corrected R1 outputs.

The old 76 files are preserved, with per-file hashes and source job/commit
metadata, under
`results/basis_a3_raw_cond_scope_depth/quarantine/superseded_noncanonical_mask/`.
They are excluded from accepted aggregates. The R2 selection has **NO
SELECTION IMPACT** because `selection/r2_layers.json` records CS-Dialogue R1
only as its source; the frozen encoder selections L2/L16/L21 remain unchanged.

Post-repair validation finds 384 Raw R1 records, 84 Raw R2 records, and 72
Conditioning records, with no duplicate accepted keys. POI metric consistency
passes for all accepted records.
