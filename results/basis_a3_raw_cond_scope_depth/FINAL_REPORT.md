# BASIS-A3 — Raw/Conditioning Scope × Depth × Cross-Corpus

Frozen/no-training explanatory study. Oracle-local results are diagnostic upper bounds, not deployable inference.

## Implementation

Exact encoder/decoder hooks, NormPreserve, free decoding, cached decoder encoder outputs, resumable condition blocks, provenance manifests, CPU audit and geometry analysis were implemented in `experiments/basis_a3.py` and `src/csasr/lss/encoder_sites.py`.

## Frozen Protocol

Raw only on all 32 encoder and 32 decoder layers at rho=.5 in R1; selected R2 layers use rho=.25 and 1.0. Conditioning is decoder-only at L24/L26/L27/L31 and rho in {.25,.5,1.0}. Global edits every eligible position/frame; Oracle-local edits only reference-aligned embedded-English positions/spans.

## Encoder Site Validation

The implemented site is the post-self-attention residual `q_enc + u_enc`, immediately before encoder FFN. The focused CPU tests passed (5/5), MIG preflight job 51626 passed worker equivalence, and real-model acceptance job 51680 passed rho=0 identity, direct NormPreserve, padding exclusion, target-union local-mask correctness, and no-gradient checks. NormPreserve was evaluated on the bfloat16 inference path with the established 1e-2 relative tolerance; maximum observed relative error was 0.0049583.

## Evaluation Panels

CS-Dialogue: 300, `sha256:7fb612ee1be7806c624dd0836fbf5e43429d41e3af032e0d4cb7fe4e8a3398ff`; dev-man: 50, `sha256:3f49d08d66e88d4bf9f79ef8a80f6013fb1d1d16e07269fb50bc94e4f26e2a3c`; dev-sge: 50, `sha256:ebdcdd10b6f0d4a5f7666097f96168bf39869a65e39de779dfdc7c8ebfb77238`. The SEAME canonical fixed panel had only 50 eligible examples per split; no outcome-based resampling was done.

## Direction Provenance

Decoder Raw/Conditioning: `sha256:5f510df5e25dc6d76976433d1376244b58ad687fe3ddc4ceb2a56f5bbe6b03d5` source artifact. Encoder Raw: `sha256:d7504dff518b16051a8cf567335fdc397eb386c4a411535e0c15b39e76ae14fc` manifest; each layer has its own D-construct artifact, unit normalization and SHA256.

## Metric Audit

`poi_corrections` and `poi_corruptions` are correctness-flip POI transitions; `poi_net_utility = corrections - corruptions`. CS candidate-level utility and outside harm were reconstructed separately with the frozen DG-03 existing_ctc candidate population. SEAME outside harm is N/A because no accepted candidate/POI population was available.

## GPU Execution

All scientific GPU work used MIG, one H100 3g.40gb allocation per job, two independent batch-1 workers, and sbatch. Valid jobs: 51638 (CS enc R1, 2:34:51), 51639 (CS dec R1, 2:05:36), 51649 (man enc R1, 15:22), 51650 (man dec R1, 10:03), 51651 (sge dec R1, 10:03), 51652 (sge enc R1, 13:23), 51654 (man R2+Cond, 11:46), 51655 (sge R2+Cond, 10:15), 51653 (CS R2+Cond, 2:00:58). Invalid preliminary attempts were quarantined and excluded.

## Raw Full-Depth Map

384 aggregate rows across three datasets. CS decoder Raw becomes strongest in late depth (Global best net: L30 (net POI 151)); encoder Raw is weak/negative on CS. Dev-man shows a small decoder response; dev-sge decoder effects are smaller. Required tables and figures are under `tables/` and `figures/`.

## Raw Encoder — Global vs Local

Corrected SEAME results sharpen the scope conclusion. Global encoder steering is negative across all three R1 panels at rho=.5. Oracle-local steering produces a narrow positive PIER region at dev-man L27-L28, including a positive L27 transition balance (7 corrections, 3 corruptions), while dev-sge has no positive PIER layer. Localization nevertheless reduces collateral damage and improves retention relative to Global, especially at late dev-man layers and across dev-sge. Local edits are much sparser, and frame-normalized efficiency is reported in the raw tables.

## Raw Decoder — Global vs Local

Late decoder Raw is powerful but collateral-sensitive on CS: L27 R1 Global has much worse matrix retention and larger outside harm than Local, while Local preserves matrix output substantially better. Dev-man shows the same direction of Global-to-Local retention improvement; dev-sge is weaker.

## Raw Dose Characterization

R2 selected encoder layers remain L2/L16/L21; decoder anchors were L0/L24/L27 plus L2 negative control. rho=.5 was reused from corrected R1; only .25 and 1.0 were decoded. Corrected SEAME R2 remains broadly negative for PIER at the selected encoder layers; dev-man L21 at rho=.25 has a small MER improvement but not a positive PIER gain. The selection remains frozen because its source was CS-Dialogue R1 only.

## Conditioning — Global vs Local

CS Conditioning L26/L27 remains modest at rho=.25/.5 but becomes damaging at rho=1. L31 is destructive globally at rho=.5/1 and substantially less damaging when local; dev-man and dev-sge reproduce the destructive high-dose L31 pattern, while L26/L27 transfer is weak.

## Encoder vs Decoder

The clearest Raw correction power is late decoder depth on CS-Dialogue. Corrected encoder evidence is weaker and corpus-dependent: a narrow dev-man Oracle-local late-layer response does not transfer to dev-sge. Decoder effects remain stronger, but also more damaging when global.

## Localization Effect

Oracle localization generally reduces matrix damage and outside harm, and rescues some late decoder conditions. It is a causal diagnostic only because it uses reference-aligned CS spans.

## Cross-Corpus Consistency

| Observation | CS-Dialogue | SEAME dev-man | SEAME dev-sge |
|---|---|---|---|
| Raw encoder useful region | none at rho=.5; late decoder signal dominates | narrow late Oracle-local L27–L28 PIER response | no positive PIER region; local reduces damage |
| Raw decoder useful region | late, especially L27–L30 Global/Local tradeoff | small late response | small L24/L27 response |
| Raw L27 correction signal | decoder clear; encoder collateral-limited | encoder Oracle-local positive PIER (7 corrections, 3 corruptions) | encoder local has no positive PIER (0 corrections, 7 corruptions) |
| Raw Global→Local rescue | clear on late decoder | strong L27 rescue versus Global, narrow benefit | retention rescue without positive PIER |
| Cond L26 useful | modest, narrow-dose | weak | weak |
| Cond L27 useful | modest Local at rho=1 | weak Local at rho=1 | absent Local |
| Cond Global→Local rescue | strong at L31 | strong at L31 | strong at L31 |
| Cond L31 destructive | yes at rho≥.5 | yes at rho≥.5 | yes at rho≥.5 |

## Qualitative Cases

See `qualitative/cases.md` and `qualitative/cases.json`; cases were selected after freezing for explanation only.

## Figures

Raw depth and repaired Raw scope/dose figures are under `figures/`, including the SEAME encoder-local updates. Geometry remains reused from the validated 32-layer package. Conditioning figures are under `figures/conditioning/`: `pier_by_layer_rho.png`, `mer_by_layer_rho.png`, `matrix_retention_by_layer_rho.png`, `correction_damage_frontier.png`, and `cross_dataset_conditioning_comparison.png`.

## Raw–Conditioning Decoder Geometry

Cosine range -0.0869..0.1472; angle range 81.533..94.985 degrees; unit-L2 range 1.3060..1.4744. L24 is among the most separated layers; L31 is the most aligned, while L26/L27 are near-orthogonal and not uniquely unusual. The unit-L2 identity was validated. Four-layer-per-dataset descriptive correlations are stored in `geometry/steering_geometry_correlations.csv`; they are not causal evidence.

## Scientific Takeaway

Raw contains real late-decoder causal correction signal on CS-Dialogue, but global application is the dominant collateral-damage problem. Encoder Raw is not yet a stable cross-corpus result. Conditioning has a narrow late-decoder response and a reproducible destructive high-dose L31 control; its positive L26/L27 signal transfers weakly.

## Risks / Negative Findings

SEAME panels are 50 rather than 300; SEAME outside harm is unavailable; oracle-localization is not deployable; geometry/effect correlations have n=4 decoder layers per dataset/scope and must not be overinterpreted.

## Next Recommendation

Do not run another study in this ticket. If work resumes, first obtain a pre-registered accepted SEAME candidate/POI accounting population before making outside-harm claims.

## Data Exposure

BASIS-A3 did not read D-dev-confirm or D-test. CS used only D-construct for directions and D-dev-select for evaluation; SEAME dev-man/dev-sge were exploratory development panels. This is an A3-specific statement: the repository's broader project exposure record reports prior D-dev-confirm exposure and DG08 D-test exposure, so this report does not claim those splits were globally untouched.

## Provenance

R1/R2/Conditioning tables: `tables/`; geometry: `geometry/`; spec freeze self-hash `8def7ba84050103343e8b94332d287258dd3497beece142738cc26e92f2fb5d3`; R2 selection: `selection/r2_layers.json`; summary: `summary.json`; repair protocol: `repair/REPAIR_PROTOCOL.json`; repaired jobs: 51682 and 51683; superseded outputs: `quarantine/superseded_noncanonical_mask/`; final manifest: `POSTRUN_ACCEPTANCE_MANIFEST.json`.

## Protocol Chronology

BASIS-A3 was initially executed across multiple committed snapshots: CS R1 at `5a5f22b`, SEAME R1 at `d4012bd`, and R2 plus Conditioning at `aa71f21`. The reconciliation found a scientific-semantic change in the SEAME encoder Oracle-local span construction: the later code uses the union of target segments with the frozen 80 ms handling, while the earlier code used the second segment boundary. The mismatch is retained in `docs/current/BASIS_A3_PROTOCOL_RECONCILIATION.md`. The exact 76 affected cells were then harmonized under the target-segment-union definition in repair jobs 51682 and 51683; all unaffected results were preserved.

## Encoder Acceptance

Focused CPU tests passed 5/5. Real-model acceptance job 51680 passed all five checks: rho=0 identity, NormPreserve, padding exclusion, Oracle-local mask equality, and no-gradient frozen inference. Detailed evidence is under `acceptance/`; the persistent repair log is `acceptance/focused_tests_repair.log`.

## SEAME Scope

The available frozen panels contain 50 utterances for dev-man and 50 for dev-sge, not the planned 300. These results must not be relabeled as 300-utterance panels.

## Metric Semantics

`poi_corrections`, `poi_corruptions`, and `poi_net_utility` are separate fields, with `poi_net_utility = poi_corrections - poi_corruptions`. Candidate-level utility, where available, remains separately named and is not used as the POI headline. The final consistency audit covers 540 accepted result records and is stored at `metric_audit/final_metric_consistency.json`.

## Post-run Scientific Repair

The SEAME encoder Oracle-local mismatch arose because early R1 outputs used the second segment boundary, while the later frozen implementation uses the union of all target segments. Exactly 76 configurations were rerun: 32 R1 layers plus six R2 dose cells for each of dev-man and dev-sge. The target-segment-union mask and 80 ms tolerance were unchanged from the canonical later implementation. All unaffected results were reused; superseded outputs remain under `quarantine/superseded_noncanonical_mask/` with source commits, Slurm IDs, and hashes. Corrected outputs feed the final tables, plots, summary, and report. R2 rho=.5 remains corrected R1 reuse.

## Limitations

- SEAME dev-man and dev-sge are small 50-utterance exploratory panels.
- The study is exploratory/developmental, not a confirmation result.
- Oracle-local steering uses reference alignment and is not deployable inference.
- A documented post-run scientific repair was required; final results carry repair provenance rather than pretending to be one untouched execution snapshot.
- No A3 output used D-dev-confirm or D-test; broader project exposure is governed by `docs/current/DATA_EXPOSURE.md`.

## Acceptance Status

`ACCEPTED_WITH_REPAIR_PROVENANCE`. The repaired grid is internally harmonized, complete, and metric-consistent. The historical mismatch and superseded outputs remain preserved for auditability.
