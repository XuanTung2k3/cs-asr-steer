# BASIS-A4 Final Report

## Execution Status

COMPLETE. All frozen Whisper and Qwen cells passed CPU completeness and provenance validation.

## Implementation

Implemented exact Whisper and Qwen pre-FFN residual sites, norm-preserving steering, deterministic official Qwen loading, frozen masks, resumable cell paths, manifests, and CPU aggregation.

## Acceptance Tests

CPU acceptance: PASS. Qwen MIG preflight: PASS (job 52090; 3.86 GiB peak VRAM; deterministic; rho=0 identity; cache identity not accepted, so canonical cache mode is OFF; no gradients). Whisper exact-site real acceptance was reused from the accepted A3 gate. Conditioning-Avg was not duplicated because the A4 redundancy gate is frozen.

## Slurm Jobs

Whisper Conditioning jobs: 52091 (CS-Dialogue), 52092 (SEAME-dev_man), and 52094 (SEAME-dev_sge). Qwen preflight: 52090; direction construction: 52136; baselines: 52137-52139; atlas: 52141-52146 (with 52140 recorded as the failed pre-fix encoder attempt).

## Whisper Completeness

Raw: 384 reused cells. Conditioning: 192 cells (24 reused, 168 newly decoded). Conditioning-Avg: 0, redundant.

## Qwen Completeness

Raw: 312 cells (144 audio-encoder, 168 text-decoder). Conditioning: 168 text-decoder cells. Baselines: CS 300, dev-man 50, dev-sge 50.

## Raw Results

Whisper MER range: (0.22847977187667082, 2.4408376963350786); Qwen MER range: (0.06000118814233945, 0.48900523560209425).

## Conditioning Results

Whisper MER range: (0.2386383888789877, 2.6); Qwen MER range: (0.06053585219509297, 0.4157068062827225).

## Conditioning-Avg Results

Not run or constructed: the frozen A4 specification proves its all-content population is redundant with Conditioning.

## Global vs Local

Whisper Global-vs-Local values remain frozen. Corrected Qwen mean MER(Global)-MER(Local), recomputed from the repaired local rows, are: Raw encoder: +0.024795; Raw decoder: +0.004664; Conditioning decoder: +0.005665. Positive values indicate lower MER under Oracle-local; no local benefit is claimed unless supported by these corrected rows.

## Encoder vs Decoder

Whisper Raw encoder steering is broadly damaging (mean layer MER-gain range -0.921 to -0.138), while Raw decoder steering has a narrow useful region (best mean layer 16, +0.013). Qwen Raw encoder effects are much smaller and mostly damaging (best mean layer 19, -0.0006); Qwen Raw decoder has a narrow best-MER region around layer 22 (+0.003). Conditioning is decoder-only: Whisper's best mean MER layer is 3 (+0.0035) while late layers can show corrective power with damage; Qwen's best mean MER layer is 24 (+0.0020), while its best mean PIER layer is 10 (+0.0054), illustrating the correction-versus-damage tradeoff.

## Cross-Corpus Findings

The CS-Dialogue, SEAME-dev_man, and SEAME-dev_sge traces are retained separately. The same depth conclusions are summarized across panels only as averages; SEAME panels remain small and no pooled generalization claim is made.

## Cross-Model Findings

The repaired Qwen local masks are no longer a no-op (CS alignable encoder/decoder nonzero rate is 100%); corrected Global-vs-Local behavior is reported numerically above. Whisper encoder steering remains substantially more damaging than Qwen encoder steering. Relative-depth patterns, not layer numbers, are compared. A high POI correction score with a negative MER gain is labeled high corrective power/high damage rather than best. rho=.5 is not treated as physically dose-matched, and vector coordinates are not compared across models.

## Geometry

Within-model Raw↔Conditioning cosine, angle, raw L2, unit L2, and vector norms are in geometry/raw_conditioning.{json,csv}; the unit-L2 identity holds for all 60 rows.

## Figures / Tables

Required depth, retention, global/local delta, decoder comparison, cross-corpus, geometry, and normalized-depth figures are in figures/. CSV tables are in tables/.

## Data Exposure

Directions use D-construct only. Evaluation uses the frozen CS, dev-man, and dev-sge panels. D-dev-confirm and D-test were not intervened; no D-test output was generated.

## Final Commit

Path-limited commit recorded after this report and its manifests were validated.

## Problems / Caveats

SEAME panels are 50 utterances each and remain underpowered for broad generalization claims. Qwen's frame grid and physical perturbation scale differ from Whisper. Conditioning-Avg remains a documented redundancy, not a missing experiment.

## Gate

READY_FOR_INDEPENDENT_AUDIT
