# SRC-CF0 — pre-run design blocked

**Status: `SRC_CF0_DESIGN_BLOCKED`. Neither scientific phase is authorized.**
This is a design-review record, not a completed experimental freeze or a scientific negative
result. `configs/inference_cf/src_cf0.json` explicitly disables execution and states that the
scientific gates are not frozen. No `PASS_TO_SRC_CF0` is issued.

Initial local and remote HEAD: `da10689870ef44ee2a382c25a4483a3d125895d6`.
Working branch: `feature/inference-cf-steering`; publication target: `origin/cs-asr-steer-inf`.
No pretrained-model forward, direction construction, intervention, reference-token evaluation,
or GPU job was performed in this review.

## 1. Scientific hypothesis and fixed scope

The requested new hypothesis is a same-prefix acoustic counterfactual representation contrast:

`delta_AC = float64(h_clean) - float64(h_English_region_masked)`

`v_AC = delta_AC / (||delta_AC|| + epsilon)`.

Both states must use frozen Whisper-large-v3, forced-ZH transcribe prompt
`[50258,50260,50360,50364]`, identical original baseline content `[:t]`, absolute query
`4+t-1`, and the native DG-02 post-cross-attention residual before FFN. Requested layers
are zero-based `[16,24]`; conditional B doses `[0.15,0.30]` and signs `[+1,-1]` give eight arms.
No other direction, teacher, acoustic perturbation, training, gradient, location optimization,
or free continuation is authorized. This supporting study does not change METHOD_CONTRACT/v6.

Construction feasibility would not demonstrate lexical grounding. Single-pulse token corrections
would not demonstrate word recovery, transcript improvement, or deployability. R0, S1,
ST-LOC0, ST-PROMPT-R1, P2-DIR and LAC0 terminal conclusions remain unchanged.

## 2. Verified panel and existing eligibility ceiling

The exact S1/ST-PROMPT-R1/P2-R panel has 180 positions, 80 utterances and 20 dialogues:
60 EN-confusion, 60 EN-correct and 60 ZH-correct. No replacement or selection is performed.
`SRC_CF0_PANEL.json` copies original query/utterance order, prefix/source metadata, runtime
membership and audio hashes, and fingerprints the sealed S1 region records. Historical strata
membership is used solely for this offline coverage review; acceptable-token sets,
reference strings and timing proxies do not enter that review. An isolated existing P2-R CPU
regression rebuilds its historical population from already-exposed evaluation inputs solely
to check equality with the old frozen population; it supplies no SRC-CF0 design features or outcomes.

Counts below were recomputed from all 80 sealed S1 candidate JSON records, joined to existing
panel membership by exact UID, t and absolute query. They are historical eligibility metadata,
not new SRC-CF0 construction or pulse outcomes.

| Stratum (original denominator 60) | Target queries | Target dialogues | Paired queries | Paired dialogues |
|---|---:|---:|---:|---:|
| EN-confusion | 28 | 12 | 20 | 9 |
| EN-correct | 39 | 17 | 35 | 16 |
| ZH-correct | 6 | 6 | 4 | 4 |

The primary masks are usable at 73/180 queries; paired controls at 59/180. Thus 107 queries
have no primary target, and 121 have no full pair. Both layers share this region eligibility;
adding a second layer, another sign or dose does not create independent safety observations.

## 3. Precise blocker

The requested causal experiment includes adequate correct-Mandarin exposure and evidence that
the vector does not damage correct Mandarin decisions. Under the unchanged S1 policy, only
six such states could receive a target intervention; only four could receive the matched
off-target comparison. Numerical/tangent/reachability guards can only decrease those ceilings.

Keeping the full /60 denominator is necessary for unconditional deployment-opportunity accounting,
but it cannot turn 54 mandatory no-edits into evidence about active-intervention safety. The
suggested allowance of two Mandarin corruptions would permit damage on 2/6 target opportunities
(33%) or 2/4 paired opportunities (50%). A single event changes those conditional rates by
16.7 or 25 percentage points. Even zero damage on all four paired states would be weak safety
evidence; multiplicity-adjusted dialogue resampling cannot create additional exposed states.

This is a design judgment about insufficient safety opportunity, not a newly imposed historical
gate, a claim of a minimum universal sample size, or evidence that v_AC fails. EN-confusion
coverage alone is sufficient for the suggested 12-query / six-dialogue construction screen,
but the complete correction-and-preservation question is not defensibly resolved by these masks.

The user explicitly requires DESIGN_BLOCKED if the fixed masks cannot support an interpretable
screen. Therefore no construction run is used to discover this already-known limitation, and
no numerical or terminal gate is tuned to make four safety opportunities appear sufficient.

## 4. Preserved counterfactual and energy semantics

The only permissible region policy remains sealed S1: R0 primary predicted EN track, original
M query attention from the ten frozen alignment heads, heard mass >=0.50, target association
>=0.10, EN region >=4000 samples, target cap 32000 samples, RMS >=1e-4, and the exact deterministic
crop ordering. Off-target is disjoint, same duration, predicted EN fraction <=0.10, attention
<=min(0.10,target_mass/2), RMS >=1e-4 and waveform energy ratio in [0.5,2].
Hard positive float32 zero on half-open [a,b), no taper, outside waveform bytes identical,
heard horizon min(N,480000), whole feature/encoder recomputation, cold model-owned M cache.
No oracle mask, low-association fallback, changed crop or absent-region substitution.

If a separately authorized design becomes viable, reuse the P2-R `solve_scale` / `scaled_direction`
and frozen `apply_steering`: target chord eta times the float64 norm of the native bf16 residual,
CPU float64 geometry, at most eight native repair emulations, geometric unreachable -> zero edit.
Historical eta and squared-energy tolerance 0.02, solver/hook 1e-6 scaled tolerance, and consumed
versus proposed relative norm tolerance 0.005 must be distinguished from descriptive bf16
NormPreserve norm-rounding. No new norm-preservation validity threshold is added here.

## 5. Unfinished freeze and next decision

The following remain deliberately unfrozen: adequate active safety coverage, construction
specificity/stability/tangent gates, complete statistical hypothesis family, causal selection
logic and terminal precedence. No requested terminal scientific label is assigned: no phase ran.
The requested eight-arm grid and prior corruption suggestions are retained as requested scope,
not operational authorization. No CF1 recommendation follows.

A human scope decision is needed before a new complete pre-outcome freeze: either explicitly
accept a narrower, opportunity-limited descriptive question with no broad Mandarin-preservation
claim, or separately authorize a change capable of supplying more active safety opportunities.
This review chooses neither, changes no masks, proposes no favorable replacement positions,
and accesses no held-out data. Preserve this blocked record under any later amendment.
