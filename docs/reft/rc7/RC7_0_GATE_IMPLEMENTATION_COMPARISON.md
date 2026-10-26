# RC7-0D — Source-grounded comparison: historical G5, historical L4, RC7 candidate

Sources read (not report text): `reft/a5000_campaign/methods.py` and `worker.py` at **2ca3c2d** (G-gates, run
core7h-20261020-v2), `reft/a5000_lra_sprint/method.py` and `runtime.py` at **b18dd0c** (L-gates, run
lra-gate-20261021-v2), and configs `configs/reft/a5000_campaign/multigpu_v2.json` and `configs/reft/a5000_lra_sprint/sprint_v1.json`.
Line numbers refer to those commits; both files are unchanged on `research/rc7-0-20261026`.

## 1. Shared allocation operator: identical to the RC7 formula

`methods.py:23–47 energy_coefficients(q, block_energy, counts, epsilon, alpha, normalize)`:

- `u = 1 + epsilon*tanh(q)` (l.37), where E = Σ valid block energies and D = Σ u²E_b (l.38–39);
- `a = alpha*u*sqrt(E/D)` (l.41–44), with zero-energy rows mapped to the scalar α and padding forced to 0;
- block energy `E_b = Σ_t m_t‖d_t‖²` in float64 (`methods.py:82`, `method.py:95`).

This is exactly the RC7 operator `a_b(α,c) = α·u_b(c)·sqrt(E/D(c))` at **c = 1**. The RC7 defaults α = 1, ε = 0.25 and 1 s
(50-frame) blocks equal the historical settings (`multigpu_v2.json:100–101`; `sprint_v1.json:42–43`). The parameter c
is a new *evaluation-time* knob: c = 0 gives the uniform parent, and c ≠ 1 gives a dose sweep of the same learned gate. It is
not a new mechanism; the historical scalar controls G1/L1 (α grid 0…2) already probe the global-scale axis, and
both selected α = 1.

Normalized arms: G4, G5 (`methods.py:117`) and L3, L4 (`method.py:103`); unnormalized: G3, L2.

## 2. Component matrix

| Component | Historical G5 (`Allocation`, kind G5) | Historical L4 (`LRAAllocation`, arm L4) | RC7 candidate | Scientifically distinct? |
|---|---|---|---|---|
| Current-state features | yes: projected block means (2) and variances (2) of h·Rᵀ, state log-RMS (`methods.py:79–80, 91–92`) | yes: state log-RMS, ‖block mean‖, √variance, within-block velocity, projected-state mean norm and variance in the update basis (`method.py:82, 92`) | proposed | **No.** Present in both |
| Neighbour-block contrasts | yes: left/right first differences of projected means, ±1 block (`methods.py:86–90`) | yes: norms of ±1-block state-mean differences (`method.py:69–72, 82–83`); L2/L3 add neighbour cosines | proposed ("contextual variation") | **No** |
| Multi-block predictor residual ν_b (EMA ρ = 0.8, train-variance normalized) | **absent** | **absent** | proposed | **Partly.** With ρ = 0, ν_b = (s_b − s_{b−1})/σ is the z-scored left contrast already in G5 (and its norm in L4). ρ > 0 extends the lag-1 difference to an exponentially weighted causal history (≈5-block effective window). This is a one-hyperparameter generalization of an implemented feature. |
| Edit norm / geometry | yes: update log-RMS (`methods.py:83`), state–update cosine (`l.85`) | yes: update log-RMS, frame state–update cosine, cos(mean state, mean update), update coherence ‖mean v‖/RMS v (`method.py:91–93`) | proposed (actual update magnitude, state–edit alignment) | **No** |
| Update-direction variation | no | yes: ‖Δ block-mean update‖ to left/right neighbours (`method.py:72, 93`) | proposed (neighbouring edit-direction changes) | **No** (present in L4) |
| Feature normalization | z-score on T2000 valid blocks with warm adapter fixed (`methods.py:96–106`; `worker.py:259–269`) | same procedure on T2000 (`runtime.py:115`) | ν normalized by v_train + δ; other features unspecified | **No**, unless v_train differs from the existing per-feature z-score. Not specified. |
| Temporal receptive field | own block + ±1 neighbour; non-causal (whole utterance encoded first) | own block + ±1 neighbour + utterance-duration fraction; non-causal | causal EMA history (all previous blocks, decaying), plus whatever current/neighbour features are kept | **Partly**: the only structural difference |
| MLP architecture / capacity | 16 → 16 tanh → 1, zero-init output, 289 params (`methods.py:65–66`) | identical (inherits `Allocation`, `method.py:48`) | unspecified | Unknown; must be matched (289) to be a feature ablation |
| Frozen vs joint training | parent frozen, gate only (`methods.py:166–167`; `worker.py:260`) | parent frozen (`runtime.py:98`) | frozen (implied) | **No** |
| Parent adapter | LoReFT CE, seed 42, lr 5e-4, epoch 3 (`a4719921…`) | DiReFT CE, seed 42, lr 5e-4, epoch 3 (`c531a22a…`) | DiReFT CE | **No**: same parent as L4 |
| Training / selection budget | AdamW lr 1e-3, 1 epoch = 2,000 examples / 125 updates, epoch selection on V300 (`multigpu_v2.json:105–106`) | AdamW lr 5e-4, 125 updates, final checkpoint only (`sprint_v1.json:23–25`) | unspecified | Must equal L4's to be comparable |
| Allocation operator | `energy_coefficients`, ε 0.25, α 1, 1 s | same | same at c = 1 | **No** |
| Gate inputs | audio-side h, d, projection only; no references, decoder states or token labels | same; invariant basis of [BA \| Bc] (`method.py:17–27`) | same (implied) | **No** |

## 3. What the archived runs already measured

| Contrast | ΔCS-MER (pp) | Allocation std | Frozen minimum |
|---|---:|---:|---:|
| G5 − G4 (adds edit-norm and alignment features to the same gate) | −0.054 | 0.0104 vs 0.0090 | 0.5 pp |
| G5 − G8 (gate vs equal-budget ungated continuation) | −0.075 | — | — |
| L4 − L3 (swaps 8 state statistics for 8 edit-geometry and update-variation statistics) | −0.011 | 0.0106 vs 0.0083 | 0.5 pp |
| L4 − L0 (gate vs equal-update ungated continuation) | −0.043 | — | — |
| best global α (G1, L1) | 0 (α = 1 selected) | 0 | — |

Every learned gate stayed close to uniform: the coefficient std was 0.008–0.014 inside the ±0.25 envelope that
ε allows. The largest gate-related gain observed is about 7–10× smaller than the screening threshold. Both feature-family swaps
already tested (state-only → +edit features in N2; state tail → edit-geometry tail in the sprint) moved CS-MER by
≤ 0.054 pp. An EMA residual is a third swap within the same operator, parent, capacity and budget.

## 4. Unspecified RC7 elements (admission blockers, not filled in here)

1. The quantity s_b being predicted: block-mean state, projected mean, update, or a scalar summary. Also its dimension.
2. v_train and δ: per-dimension or pooled, and whether computed with the parent fixed on T2000 valid blocks.
3. Feature set: ν added to, or replacing, the L4 features. The total must stay at 16 inputs / 289 parameters to match.
4. Initial predictor state ŝ_1⁻ and its treatment at utterance start and padding.
5. Optimizer, lr, updates and checkpoint rule (L4 used lr 5e-4, 125 updates, final checkpoint).
6. Whether c is fixed at 1 or swept; any sweep is selection on V300 and must be counted.
