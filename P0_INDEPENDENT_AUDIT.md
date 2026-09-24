# P0 independent audit — inference-time counterfactual feasibility

**Auditor role:** independent re-verification of the committed P0 result (jobs 54702/54703)
before any P0-R1 code change. Nothing in the prior Codex summary or in
`experiments/inference_cf_p0_aggregate.py` was trusted as evidence; every scalar below was
recomputed from the immutable row shards with a standalone script that imports **no** `csasr`
code (`states`, `real_scores`, `shuffled_scores` re-reduced with numpy).

- Worktree `/home/tungnx/cs_asr_steer_inf`, branch `feature/inference-cf-steering`, audited at
  HEAD `e4bc24b`.
- Frozen spec `docs/inference_cf/P0_FEASIBILITY_SPEC.md` (pre-run freeze `55faf76`).
- Corrected run: job **54703**, `results/inference_cf/p0_retry/`, manifest
  `sha256:0287abb2f6537f1d88f179fa62310424e411b1baee47258670c87d781badfb6c`.
- Preserved failed attempt: job **54702**, `results/inference_cf/p0/`, manifest
  `sha256:2ef71c48…` (distinct hash → clean version separation).

## Verdict

The independent recomputation **reproduces every headline number** in the Codex report and the
committed `summary.json`, to numerical precision. All three gate verdicts are confirmed:

| Gate | Reported | Independently confirmed | Basis |
|---|---|---|---|
| G1 | `DEGENERATE_BY_CONSTRUCTION` | **CONFIRMED** | `c0`==`cM` token IDs `[50258,50260,50360,50364]`; `cE`=`[50258,50259,50360,50364]`. `c0` and `cM` state vectors are **bitwise identical** on all 57 kept rows ⇒ `‖h0−hM‖ = 0` exactly (not merely ≤0.02). `‖hE−hM‖` median 2.749 (range 1.058–7.744); `rho` ≈ −1; `direction_norm` ≈ 1. |
| G2 | `PASS` | **CONFIRMED** | 57/60 retained, 3 documented skips. **0 alignment violations** across all kept rows: shared logical position, full-replay masks all-ones, `cache_positions=0..L−1`, `cache_used=false`, `beam_lineage=greedy:0`, L24 site, query index = prompt_len+logical−1, 1280-d states, `shared_prefix==content[:logical]`. |
| G3 | `WEAK/BLOCKED` | **CONFIRMED** | Finite coverage 57/60 = 95%; candidate collision 50/57 = **87.7%**; usable q_cross rows: 3 wrong-English, 3 correct-English, 1 Mandarin; AUROC **not estimable** (<10/class); audio-control abs Δq_cross mean 1.994 over only 7 rows. Predeclared G3 thresholds (≤50% collision, ≥10/class, AUROC≥0.60) fail. |

Because G1 = DEGENERATE_BY_CONSTRUCTION, G2 = PASS, and G3 is blocked **specifically** by the
K=1 candidate collision rate (see MAJOR-1 for the mechanism, confirmed to be genuine and not an
implementation defect), the preconditions to enter the P0-R1 evidence-definition repair are met.

## What was recomputed independently (all match)

- **Manifest / panel / permutation hashes** re-derived from canonical JSON: all match the stored
  values; `summary.json` manifest_hash matches; **all 11 manifest source-file hashes still match
  the on-disk files** (no post-run drift of spec, executor, core, prepare, sbatch, model, or
  data-role parquets).
- **Geometry** (h0/hM/hE norms, `delta_norm`, `direction_norm`, `c0_cM_residual_norm`, `rho`,
  collapse factor): max |recomputed − stored| = 3.3e-16 over all fields, all 57 rows.
- **Support** (`collision`, `q_own`, `q_cross`) recomputed from raw scores: 0 mismatches vs
  stored `real_support`/`shuffled_support`. Collision→score-identity invariant holds
  (`yE==yM ⇒ sE_yE==sE_yM ∧ sM_yE==sM_yM`).
- **Collision rate** 50/57 = 0.8772; by stratum wrong 15/18, correct 16/19, Mandarin 19/20.
- **Audio permutation**: 60-cycle bijection `perm[ids[i]]==ids[(i+1)%60]`, no self-pairs, every
  shuffled sha differs from own sha.
- **Skips**: the same 3 `frozen_baseline_text_mismatch` identities in **both** jobs
  (`english_correct:ZH-CN_U0085_S0_167:20`, `english_wrong:ZH-CN_U0091_S0_220:15`,
  `english_wrong:ZH-CN_U0101_S0_203:51`) → deterministic, not random.
- **Existing tests** `tests/test_inference_cf_p0.py`: 7/7 pass.

## Audit findings

### CRITICAL — none.
No correctness, provenance, alignment, or data-leakage defect was found that would invalidate
G1/G2/G3. The scientific repair (Phase II) may proceed.

### MAJOR-1 — K=1 collision is a genuine property of the argmax readout (disposition: motivates P0-R1, not a bug)
The dominant G3 failure is genuinely the K=1 candidate-collision rate, **not** an implementation
defect. Mechanism, quantified from shards:
- `cM` and `cE` differ **only** in the language token (`<|zh|>`=50260 vs `<|en|>`=50259) over the
  **same** audio and **same** content prefix.
- On **45 of 50** collision rows the two conditions assign **different** log-probs to the shared
  top-1 token (median |logp_cE(t)−logp_cM(t)| = 0.0033 nats, max 0.286), and the L24 states
  differ (‖hE−hM‖ median 2.61 on collision rows). The conditions genuinely diverge; the **top-1
  argmax simply coincides** and collapses that difference.
- No collision involves EOS or a special token (0/50).
- Ruled out as causes: parser/token handling (fixed and verified, see MINOR-2), special/EOS
  behavior (0), scoring (support recomputed, invariant holds), distributional identity (distinct
  states + distinct logp for shared token).
- ⇒ A short predeclared continuation (K=3) that lets the two conditions diverge over multiple
  steps is the minimal, well-motivated repair. This is the exact `K=1→K=3` change Phase II
  authorizes.

### MAJOR-2 — the 7 usable K=1 rows are positionally unrepresentative (disposition: reinforces BLOCKED)
Non-colliding (usable) rows sit at logical-position **median 0.0**, whereas collision rows sit at
**median 21.5**. Nearly all K=1 evidence comes from first-content-token positions, where the
language prompt has maximal leverage. The current q_cross panel is therefore both tiny (n=7) and
biased toward position 0; it cannot support G3 and its audio-control statistic (n=7) is not a
panel-wide result. This does not change the verdict (already BLOCKED) but confirms the K=1
evidence is unusable, and argues the P0-R1 panel must retain the same positions unchanged.

### MINOR-1 — c0 is a redundant recomputation of cM (disposition: freeze the consequence in P1)
`c0` and `cM` are not merely token-identical; the executor runs an identical forward for both, so
their states are bitwise equal and `‖h0−hM‖=0` exactly. The original matrix-collapse "need"
factor (which requires a c0-vs-cM contrast) carries **zero** information and is non-identifiable.
Per Phase II-A this factor must be permanently removed from any P1 need signal.

### MINOR-2 — parser fix `cba7784` is legitimate and narrow (disposition: accepted)
Job 54702 failed all 57 non-text-mismatch rows with `baseline_prompt_mismatch` because the
original code assumed `generate()` returns `[prompt]+[content]`. Verified from shards: **0/57**
baseline sequences begin with the prompt or with `<|startoftranscript|>` — this installed
Transformers returns **content-only** sequences. The fix replaces the prompt-strip with
`generated_content()` (strip at first EOS) and adds a regression test. It touches **no** gate
threshold, scoring, geometry, candidate, or selection logic. The failure was a code defect
exposed before any result was read, so repairing it and rerunning is sound provenance.

### MINOR-3 — aggregate script hardened in `e4bc24b` after the run (disposition: accepted)
The final commit added assertions (audio-sha consistency, prompt-length/query-index checks,
in-script geometry recomputation) to `inference_cf_p0_aggregate.py`. Gate **logic and thresholds
are unchanged** from the frozen spec. Because this audit recomputes every number without that
script, the change does not affect the evidence.

### MINOR-4 — no EOS emitted within 200 tokens (disposition: note for P0-R1 truncation rule)
No baseline sequence contains EOS (0/57); some hit the `max_new_tokens=200` cap. Harmless for the
K=1 single-token readout, but P0-R1's K=3 continuations require an **explicit frozen
EOS/truncation rule** (Phase II-C), since a continuation may reach EOS or the content end.

## Audit coverage checklist

| Item | Result |
|---|---|
| Reference / future-token leakage | None. `inference_row` takes no reference/true_future/oracle/poi/stratum (test enforces). Candidates are `argmax` of model logits; the diagnostic token `content[logical]` is recorded but excluded from the fed prefix `content[:logical]`. |
| Dev/test separation | Only `D-dev-select` surfaces read (role parquet, POI parquet, DG-04 B0). No confirm/test/SEAME/ASCEND/oracle. All panel rows role `D-dev-select`. |
| Predeclaration timing | Spec + gates + code frozen `55faf76` (13:25) before job 54702; parser fix `cba7784` (13:27) before job 54703; report `e4bc24b` (13:31). Gate thresholds never moved after data. |
| Logical prefix alignment | `shared_prefix==content[:logical]`, `logical==len(shared)`, query index consistent, all 3 conditions share prefix. 0 violations. |
| Mask / cache positions | All-ones masks; `cache_positions=0..L−1`; `use_cache=false`; no KV exchange. 0 violations. |
| Candidate generation | `argmax` per condition on real audio only; no reference. |
| Score normalization | Single next-token `log_softmax`; K=1 so no length norm; support algebra reproduced exactly. |
| EOS behavior | 0 EOS/special candidates; 0 EOS in baselines (MINOR-4). |
| Deterministic audio permutation | Cyclic 60-bijection, no self-pairs, verified. |
| Duplicate / resume | 60 unique identities; `validated_row` guards manifest_hash/identity on resume. |
| Parser-fix provenance | Verified legitimate and narrow (MINOR-2). |
| Manifest / hash consistency | Manifest, panel, permutation, and all 11 source hashes match; two jobs carry distinct manifest hashes. |

## Conclusion

G1 = `DEGENERATE_BY_CONSTRUCTION`, G2 = `PASS`, G3 = `WEAK/BLOCKED`, and the G3 block is caused by
a genuine K=1 argmax-collision property rather than any defect. The Phase II P0-R1 precondition
(repair only the G3 evidence definition, K=1→K=3, everything else frozen) is satisfied. P1 has
not started and does not start here.
