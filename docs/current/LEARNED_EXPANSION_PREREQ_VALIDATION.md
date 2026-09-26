# Learned-Expansion Prerequisite — Frozen Layer Validation (STOP decision)

**Decision (ticket §2 gate):** the one-seed learned exploration is **BLOCKED** and has
**not** been launched. No GPU job was submitted; no result artifact was created; no
`D-test`/`D-dev-confirm` was read; no DG-00…DG-08 result was modified.

## Why the §2 gate trips
The ticket forbids selecting a learned layer from the 10-utterance atlas alone when a
larger frozen validation exists, and requires a STOP + recommendation when the full
frozen validation has **not** been run. The frozen evidence on disk:

| Evidence | Population | Layers | Status |
|---|---|---|---|
| DG-03 causal screen (`results/dg03/screen/`) | full D-dev-select (300 utt) | **L16, L24** | validated; selected L24 |
| DG-04 frontier (`results/dg04/`) | full D-dev-select (300 utt) | **L24** | validated |
| BASIS-A ablation (`results/basis_ablation_frozen/`) | full D-dev-select (300 utt) | **L24 only** | validated (R/L/Cond/RC/LC) |
| BASIS-A2 all-layer atlas (`results/basis_frozen_layer_atlas/`) | **10-utt micro-panel** | 0–31 | `EXPLORATORY — NOT FULL-DEV VALIDATED` (its own label) |

A full (300-utt) frozen validation exists **only for {L16, L24}**. Every candidate layer
the atlas nominates outside {16,24} is supported by a 10-utterance panel only, with its
own `candidate_layers_exploratory.json` status `EXPLORATORY — NOT YET FULL-DEV VALIDATED`
and noise-level Raw/Local utilities (U = +2/−1/−3 on 10 utts at L0/L1/L6).

Second blocker: there is **no executable learned-expansion training runner** (the
prep stage built modules/configs/tests but deliberately no runner; DG-05/06/07 runners
are frozen and L24/basis-hardcoded). Even the L24-anchor completion (Raw-only, Raw+Cond
adaptive) cannot be executed until such a runner is built and unit-tested.

## Frozen candidate evidence (extracted, exploratory unless noted)
- **Raw / Local:** near-identical in the panel; atlas top-utility layers {0,1,6} are
  noise-level; frozen dose-response is "damage grows faster than correction" at every
  layer/dose. No promising non-L24 Raw/Local layer in frozen evidence.
- **Conditioning:** the one notable cross-layer signal — strongest exploratory panel
  utility at **L26–L27** (L27: 20 corr / 0 corrupt, PIER-gain 0.294, matrix-ret 0.955;
  L26: 22 corr / 0 corrupt, PIER-gain 0.324), vs L24 where Conditioning is
  "DAMAGE GROWS FASTER THAN CORRECTION" (full validation). **Not a validated selection.**
- **L24 anchor:** fully validated (DG-03/04 + BASIS-A).
- **Negative control:** L31 (descriptive only).
- **Geometry (population-independent, from D-construct):** RAW+COND and LOCAL+COND span
  the **same 2-D subspace** (max principal angle 3.08e-6°, projection distance 1.08e-15);
  residualization removes ≤2.17% of normalized Raw energy — a coordinate change, not a
  change in representational content.

## Smallest required frozen validation (recommended, pre-register before running)
A full-dev frozen steering screen — **no training** — to accept/reject candidate layers
before any learned run at them:

- **Population:** full D-dev-select candidate population (the 300-utt set used by
  DG-04/05/06/07 selection), not the 10-utt panel.
- **Layers (predeclared, ≤5):** L24 (anchor/cross-check), **L26, L27** (Conditioning
  band), **L31** (negative control), and the best Raw/Local nominee (L0 and/or L6) so the
  Raw/Local layer question has a non-L24 data point.
- **Directions:** Raw, Local, Conditioning, Raw+Cond, Local+Cond (exact site; per-layer
  directions already cached under `results/basis_frozen_layer_atlas/directions/`).
- **Dose:** ρ = 0.5 (DG-04 reference); optionally the frozen {0.5,1.0,2.0} grid.
- **Decode:** greedy, temperature 0, beam 1.
- **Metrics:** canonical `result_v1` (MER/PIER/emb-WER/matrix-CER, corrections/
  corruptions/utility/outside-harm, embedded + matrix retention, energy).
- **Selection rule (pre-registered):** a non-anchor layer becomes a learned candidate
  only if its frozen screen shows non-trivial correction presence with bounded damage at
  ρ=0.5; L24 is always retained; L31 is the negative control.
- **Caveat recorded:** frozen (ungated) net utility is expected negative even where the
  *learned* controller succeeds (cf. L24: frozen Raw/Local damaging, but DG-05/06 learned
  M\* positive). The screen therefore gates on correction-presence + damage-boundedness,
  not on frozen net utility alone.
- **Provenance/GPU:** commit a `PLANNED_PRE_RUN` frozen manifest before any GPU (as
  BASIS-A did); ≤2 MIG jobs; reuse the BASIS-A2 free-decode code path scaled to 300 utt.

## Parallel prerequisite (no layer validation needed)
Build + unit-test `experiments/learned_expansion.py` wiring the expansion modules
(`build_steering_module`, `MultiSiteSteering`) into the frozen DG-05/06 training + greedy
D-dev-select decoding loop with a configurable layer, reusing `C_E`/`R_M` and the D1
objective. This unblocks the L24-anchor completion (Raw-only, Raw+Cond adaptive), which
needs no new frozen validation, and is the executable path the later multi-layer runs use.
