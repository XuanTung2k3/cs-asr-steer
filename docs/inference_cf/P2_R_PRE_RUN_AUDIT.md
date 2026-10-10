# P2-R pre-run audit

**Verdict: `PASS_TO_P2_R_RUN`.**

- **Audited commits:** `da876c7` (frozen spec set) and `d96a64d` (end-to-end test). No P2-R GPU
  outcome existed at audit time.

| Check | Evidence | Result |
|---|---|---|
| Spec frozen before outcomes | Spec, config, population, harness, analysis/decision code and Slurm committed and pushed (`da876c7`). The Codex contract is archived verbatim. | ✅ |
| Population frozen | `results/inference_cf/p2r/population.json`, hash `sha256:6a880d2b…`. It reproduces exactly from the frozen config (test). 60/60/60 positions; EN-confusion covers 17 dialogues (13 span-initial); 80 utterances; 209 D2 targets. Every stratum is estimable (≥ 40 positions, ≥ 12 dialogues). | ✅ |
| Data roles / no P3 leakage | All 80 utterances are `D-dev-select` (role manifest). Overlap with `D-dev-confirm` or `router-calib` IDs is 0; only ID metadata was read. No `D-test`, P3 or transfer file is read by any P2-R module. The runner reads only the manifest, config, population, the matched-baseline P2 rows and the R2 audio panel. | ✅ |
| Oracle separation | Reference-derived items (target token sets, D2 target lists, CTC midpoints) exist only in the population file, which the diagnostic runner receives explicitly. See the details below. | ✅ |
| Conditions fixed | cB/cE, language IDs, L16 site, NormPreserve, forced prefix 4, suppression and the 200-token cap are identical to P2. The D2 current schedule is the frozen P2 selection. | ✅ |
| Energy matching valid | Target e\* = 1.12608, taken from the P2 records. See the details below. | ✅ |
| Identity and isolation | See the details below. | ✅ |
| Metrics and decisions frozen | 11-contrast primary family, Bonferroni, dialogue bootstrap (10,000 reps, seed 240924), δ_p rules. See the details below. | ✅ |
| Random seed frozen | `sha256("P2R-random-v1\|uid\|t")`, one vector per position, orthogonal to the state, with a degeneracy fallback. The test verifies determinism and construction. | ✅ |
| No new method search | One target energy, the frozen P2 α/gate/direction for D2, and one random vector. There is no layer, α, gate, localizer, condition or direction grid. | ✅ |
| Compute | One job on `mig` with a 2-hour limit, batch 1, and a resumable ledger. The queue is empty. | ✅ |

**Oracle separation, details:**

- An AST test asserts that the deployable modules (`inference_cf_cached.py`, `inference_cf_p2.py`)
  contain no identifier or import involving p2r, target, oracle, reference, evaluation units or
  CTC.
- `cached_decode` has no plan or target parameter.
- The deployable path and its core modules are byte-identical to the P2 freeze (`43a73c7`).
- References are read only by the CPU analysis.

**Energy matching, details:**

- The solver works on the site state and the hook's own arithmetic, never on logits.
- The solver's realized edit equals the hook's audited edit bitwise (tests in fp32 and bf16).
- Unreachable arms are recorded, never relaxed.
- Matching requires a 2% pairwise and to-target squared-energy tolerance.
- D2 total energy must match within 2%.

**Identity and isolation, details:**

- Every pulse arm starts from the same unedited cache, and the cache is restored bitwise (test).
- +d and −d act at the identical state: φ₊ + φ₋ = π (test).
- The current-placement replay reproduces the deployable `cached_decode` step by step: g,
  fallback, edit and edit norm (test).
- On the real run, the analysis also verifies the replay against the saved P2 steps.
- The oracle replay executes the relocations with matched energy, and the output serializes
  (end-to-end test).

**Metrics and decisions, details:**

- The decision logic is `inference_cf_p2r_analyze.decide`, unit-tested on each profile.
- Nonsignificance is never read as equivalence.
- Invalid runs are forced to `MECHANISM_STILL_AMBIGUOUS`.

**Tests:** 86 inference-CF tests pass, including P0/R2/P1/cached/CE/P2 regressions and 12 P2-R
tests.

**Residual risks (not blockers):**

- Most EN-confusion targets lie inside translated spans. Reference next-token evidence there is
  conditioned on an already-translated prefix. It is reported by span type and does not enter the
  decisions.
- Some energy-matched arms may be unreachable at positions where d is nearly parallel to the
  state. These are counted and excluded, not relaxed.
- n = 60 per stratum limits precision. Wide intervals end in ambiguity by rule.

**PASS_TO_P2_R_RUN**
