# P0-R1 evidence spec — frozen before GPU output

Versioned revision of the P0 feasibility diagnostic. It repairs **only** the G3 evidence
definition after the independent audit (`P0_INDEPENDENT_AUDIT.md`) confirmed G1 =
`DEGENERATE_BY_CONSTRUCTION`, G2 = `PASS`, and G3 = `WEAK/BLOCKED` where the block is caused by a
genuine K=1 argmax-collision property (87.7%), not a defect. The single scientific change is
**K = 1 → K = 3**. This diagnostic performs no intervention and does not start P1.

Governing frozen documents unchanged: `docs/inference_cf/P0_FEASIBILITY_SPEC.md` (P0),
`P0_INDEPENDENT_AUDIT.md`. This spec inherits every P0 rule except the candidate construction and
scoring defined in §2–§3.

## 1. Consequence of G1 (frozen)

Because `c0` and `cM` are token-identical and produce bitwise-identical L24 states
(`‖h0−hM‖ = 0` on all retained P0 rows), the original matrix-collapse / geometry "need" factor
(`rho`, `c = max(0,−rho)`) is **non-identifiable** and is permanently removed as a P1 need signal.
No alternative `c0` condition is invented. The only P1-consumable direction, *if* evidence
becomes valid, is

```
d_{l,t} = normalize( hE_{l,t} − hM_{l,t} )   at l = 24, the DG-02 site
```

with an **evidence-only gate** derived from a validated `q_cross`. P0-R1 does **not** implement the
P1 decoder, and does not tune any gate temperature/strength.

## 2. Revised candidate construction (K = 3)

Everything below reuses the frozen P0 panel, permutation, conditions, site, model, precision, and
the shared-content-prefix definition **unchanged** (the panel and permutation are copied
byte-identically from `results/inference_cf/p0_retry` and re-verified by hash).

From the same audio `x` and the same generated content prefix `y_<t` (= `content[:logical]`,
`logical` chosen by the frozen offline fraction rule), generate **one deterministic K=3
continuation per condition** by greedy argmax under the existing frozen decoding convention
(beam 1, no sampling):

- `yE` — greedy K=3 continuation under `cE`;
- `yM` — greedy K=3 continuation under `cM`.

No reference text, future gold token, oracle switch position, target-language annotation, POI, or
stratum label enters candidate generation.

**Frozen EOS/truncation rule.** Take up to 3 greedy tokens; the candidate is the prefix **before**
the first EOS, capped at 3. A candidate therefore never contains EOS or a special token, and its
length `|y| ∈ {1,2,3}`. If either `yE` or `yM` truncates to length 0 (immediate EOS), the row is
skipped with reason `empty_continuation`.

## 3. Common-candidate cross scoring

Score **both** complete K=3 candidates under **both** conditions by teacher-forced full causal
replay (`use_cache=False`, all-ones mask, positions `0..T−1`), using **token-average log
probability** with the frozen truncation rule:

```
sC(y) = (1/|y|) * Σ_j  log p( y_j | x, shared_prefix, y_<j, cC )   for C ∈ {M, E}
```

The log-prob of `y_j` is read from `log_softmax` of the logits at absolute query index
`len(prompt_C) + len(shared) + j − 1`. Compute:

```
q_cross = 0.5 * [ (sE(yE) − sE(yM)) + (sM(yM) − sM(yE)) ]     # primary
q_own   = sE(yE) − sM(yM)                                      # diagnostic only
```

Save all four components `sE(yE), sE(yM), sM(yE), sM(yM)`, their per-step log-probs, both candidate
token sequences, and their **longest common prefix** length (logged, never used in the score).

**Candidate collision.** If the full K=3 sequences are identical (`yE == yM`), the row is a
candidate collision; `q_cross` is uninformative (identically 0) and reported `null`, mirroring the
frozen K=1 collision rule so the collision-rate metric stays comparable.

Prohibited (unchanged from P0): top-k / beam candidate pools, learned classifiers,
vocabulary/language masks, reference-derived English-token sets, alternative scorers, and
temperature tuning. P0-R1 tests exactly one repair.

## 4. Audio-dependence control

For the **same** fixed candidate sequences and prefixes, rescore all four components under the
existing deterministic cyclic mismatched-audio permutation (no self-pairs). Only the encoder /
audio representation changes; candidates are identical between real-audio and shuffled-audio
scoring. Save all four shuffled components and per-step log-probs. No second candidate generation
on shuffled audio.

## 5. Geometry continuity (G1/G2)

At the diagnostic query position, capture `h0` (from `c0`) and `hM`, `hE` (from the K=3 step-0
forwards of `cM`, `cE`) at L24, and recompute the P0 geometry (`‖h0−hM‖`, `‖hE−hM‖`, `rho`, direction
norm) to reconfirm G1 = `DEGENERATE_BY_CONSTRUCTION` and G2 alignment on the reused panel.

## 6. Predeclared decisions (frozen G3 logic, unchanged from P0)

- **G1**: `IDENTIFIABLE` / `DEGENERATE_BY_CONSTRUCTION` / `BROKEN` by exact prompt tokens and state
  residuals (≤ 0.02 L2). Expected to remain `DEGENERATE_BY_CONSTRUCTION`.
- **G2**: `PASS` only if every retained row has shared logical position, full-replay mask/cache
  positions across **all** continuation-scoring positions, L24 site, and greedy lineage; every
  non-retained row has a reason. Otherwise `BLOCKED`.
- **G3**: `SUPPORTED` only with ≥ 90% finite retained coverage, ≤ 50% full-sequence candidate
  collision, ≥ 10 non-colliding wrong-English **and** ≥ 10 non-colliding Mandarin rows, AUROC ≥ 0.60
  for wrong-English vs Mandarin, **and** mean paired absolute real/shuffled `q_cross` change ≥ 0.05
  across ≥ 10 non-colliding rows. Correct-English reported separately regardless. Otherwise
  `WEAK/BLOCKED`. **These thresholds are not weakened after seeing P0-R1.**

## 7. Run discipline

One physical GPU job on the same justified H100 MIG `3g.40gb`. No layer/K/alpha/temperature/scorer
sweep. The manifest is versioned `schema = p0r1_cross_k3_v1`, `candidate_k = 3`, and records
`supersedes_manifest = <P0 manifest hash>`. Original P0 artifacts (jobs 54702/54703) are preserved
unchanged.

## 8. Final decision

`P0_PASS_REVISED_EVIDENCE_ONLY` only if G1 remains `DEGENERATE_BY_CONSTRUCTION`, G2 remains `PASS`,
and the revised `q_cross` satisfies the frozen §6 G3 criteria — then freeze
`d_{l,t} = normalize(hE − hM)` and an evidence-only gate for P1, without optimizing gate
strength in P0. Otherwise `P0_BLOCKED_EVIDENCE`: state that the candidate-continuation support
formulation has failed P0 and a new support definition must be designed as a separate versioned
experiment; do not immediately try K=2/K=4/top-k/another scorer in the same session.
