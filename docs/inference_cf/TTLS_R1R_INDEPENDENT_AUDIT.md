# TTLS-R1R independent scientific audit

**Verdict: `AUDIT_PASS_TTLS_INSUFFICIENT`.**

**Decision: prioritize A2+P; close this version of TTLS.** Numerical repair and reproduced metrics are distinct from
method effectiveness. The repaired TTLS has a clean zero-edit start and nonzero gradients, but its best CE arm produces
four wrong-language POI corrections in one phrase on one utterance/dialogue. Eight other corrections are deletion or
spacing repairs. AC adds no canonical recognition benefit over direct substitution. This does not justify expanding
this global-vector, two-update, L16 configuration. It does not disprove representation adaptation in general.

No frozen method, historical source, result, report or shared kernel was changed. No Round 2, alternative objective,
LR/mask/layer search, confirmation/test or transfer execution was started. This audit is on branch
`audit/ttls-r1r-independent`, worktree `/home/tungnx/cs-asr-steer-ttls-r1r-audit`.

## 1. Provenance and repair-only scope

| Item | Independently verified value |
|---|---|
| Repository | `https://github.com/XuanTung2k3/cs-asr-steer.git` |
| Original completed R1 | `800fee26a55651c307a1a33b066fdb47ece22b05` |
| Original executed source | `90547f4b1dd827358cb09d0c02f2a143899adacb` |
| Original independent audit | `f99f176b60b403da282852957410298d06138ba2` |
| R1R source/spec/config freeze | `38723a01ad17a923d63a5dd36bf52d7fe4f08c87` |
| Resolved R1R manifest commit | `77e68fa5` |
| R1R output seal commit | `caa4f3c1`, before report/evaluation |
| Completed R1R revision audited | `f1573198866cdb3edd2b3b1acf3a3b08b627be82` |
| R1R job | 58391, H100 MIG 3g.40gb; 366.331 s measured work |
| Manifest hash | `sha256:6d39fafe02f7423df108a716d1054656528af639a6297dbf36e76359b52c651d` |
| Output seal hash | `sha256:6cba364127a815b74fab02eb5cc935faa6a39a36e60ec173e8c9c4bdca172a22` |
| R1R config SHA256 | `d4f378eac13d883e41df49d128269641e41278e268e410c9b83b795efe861f83` |
| Fixed100 panel SHA256 | `266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5` |
| Teacher/ID plan hash | `sha256:61ba6a5216e12e683e82ae0a8ca405cbe67a450bfa28e4c13682ef244169e31e` |
| Model weights SHA256 | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| Tokenizer JSON SHA256 | `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca` |
| Runtime | Torch 2.10.0+cu128; Transformers 4.57.6; NumPy 1.26.4; BF16 eager; greedy; cap 200; seed 240924 |

Model/tokenizer are the local `/mnt/data/tungnx/whisper-large-v3` snapshot. No upstream Hugging Face commit revision
is recorded in the production manifest; reproducibility is pinned to all eleven local file SHAs. The audit verifies
these files, all manifest source pins, all 204 seal files, all 100 audio full SHAs, exact plan identity/order and the
original R1 pinned source/output hashes. Exact100 IDs and14-dialogue replay IDs are in audit JSONs. References come
from the canonical dialogue-v2r3 D-dev-select parquet filtered to these 100 IDs; its full SHA is recorded independently.

Population: already-exposed fixed100, 20 dialogues × 5, a FULL300 subset. The enriched24 panel and D12 are descriptive
subsets, not independent samples. No new data role or held-out reference was accessed. AGENTS, METHOD_CONTRACT,
DATA_EXPOSURE, original spec/audit/patch proposal and repaired spec/report/code/raw files were inspected. This is a
separately frozen exploratory adaptation experiment, not the core-v6 basis/controller implementation.

Git history establishes freeze→manifest→output-seal→evaluation ordering. Commit history alone cannot prove historical
push timing or rule out unrecorded earlier reference access. The audit's own replay output is sealed and pushed
before any new replay-reference scoring.

### Exact numerical change

```python
# Historical, left unchanged:
steered = steered / (new_norm + EPS) * orig_norm
# Versioned TTLS-R1R only:
steered = steered * (orig_norm / new_norm.clamp_min(EPS))
```

This changes both operation ordering **and** additive-epsilon versus denominator-floor semantics. EPS remains 1e-6.
The ordinary Whisper states are nondegenerate, so the floor does not activate there. Native BF16 nonzero rounding
changes are expected. At zero edit and norm above EPS, the ratio is exactly 1. There is no zero-z branch/detach that
would disconnect the optimizer. Historical `models/hooks.py`, `lss/sites.py`, `ttls.py` and every original R1 source
pin remain byte-identical.

`EpisodeR1R` inherits fresh FP32 z, AdamW, two updates and projection from R1; only its hook-forward path changes.
Direction/site L16, prompts, teachers, candidate/stable-set rules, masks, two steps, lr=E*/(2√1280), budget and decoding
are retained. No new optimization objective was introduced. All theta0 candidate records, p0 values and stable sets
match the original raw files, not merely the subset enforced by the production boolean gate.

Recomputed: repaired T1/T2/T4/T6; clean B0; nonempty zero ALL/accepted AC controls. Reused: AUTO, historical A2,
A2-CE+P, A2-AC/+P, CD, ACSUB, original TTLS contrasts, and historical zero-vector census. Their original seal verifies.
The earlier independent audit already reproduced matched LN arms and archived A2 states; none uses the repaired hook.

Recorded non-actuator departures: stronger integrity instrumentation; detached scalar logging in new loss parts;
a new frozen lexical taxonomy and stricter promising rule. The inherited `Episode.run` still converts scalar losses
without detach and emits the old warning; the logging change does not detach learning. The stricter decision rule was
frozen before repaired outcomes but informed by already-exposed R1 evidence; it is not a confirmation criterion.
Only STATUS/CODE_MAP received additive historical notes; the original contracts/results were not rewritten.

## 2. Independent numerical verification

The audit independently recreates the old BF16 failure with a nonempty mask and verifies the new zero identity.
CPU checks derive the analytic gradient at z=0:

`Σ_t g_t [w_t − u_t(u_tᵀw_t)]`, with `u_t=h_t/||h_t||`.

The actual repaired-kernel gradient agrees in float64 to 1e-12, directional finite differences to 1e-7; native BF16
zero identity keeps a finite nonzero z gradient. These checks do not import Claude's assertions. A second independent
model-site injection implements its own ratio expression and full cross-attention residual reconstruction, including
**both FFN and its residual skip**. A normalization-input-only hook is not an equivalent gradient check.

The saved100 integrity rows are checked from raw token/text/termination arrays, not only their pass flags. Every
ALL zero decode equals the clean historicalB0, all 6 AC masks also match; all zero logits/FFN flags show exact identity,
initial preservation KL is0 and every episodic budget/reset/consumption probe satisfies its frozen bound.

The independent GPU check and compact replay are described in §3. All-parameter bytes and no-grad/reset/hook
ownership are verified. TTLS trains only a fresh 1280-scalar activation vector; all Whisper weights stay frozen.
A2's 194 decoder-LN tensors/248320 FP32 masters and settings remain unchanged and separately audited in R1.

Raw full100 optimization: T1 mean CE .413072→.343246→.293638,100 active episodes; T2 also100. AC accepts6 and abstains94.
Both updates have finite nonzero gradients; projection never binds. Largest consumed chord1.094216; maximal FFN
relative norm-rounding error.008402, below .02. Zero direct displacement outside the declared mask holds on the fixed
teacher history. This local result does not constrain indirect autoregressive consequences after a changed token.

### Minor mechanism/reporting correction: KL value 0 is not an exact gradient-free assertion

R1R correctly fixes **initial KL value** to0. However, step 0 total-gradient norms differ between T1/T2 in 83/100 rows
(max absolute.000252373, relative.001851) and T4/T6 in 5/6 rows (max absolute.00358359, relative.001662). Independent
pure-KL gradient probes quantify the small finite-precision residue. Equal float32 log distributions do not imply an
exactly zero numerical backward through exp/log-softmax and the shared native BF16 graph.

Thus “P is exactly gradient-free / can act only on update2” is an ideal-real-arithmetic statement, not a bitwise fact
about this run. The effect is small, the declared scalar-KL tolerance is satisfied and the same frozen objective is
used. This is a mechanism/reporting correction, **not** a recurrence of the zero-start defect or grounds to invalidate
the repaired hypotheses. No metric-driven loss repair or rerun is proposed here.

Production phase-A enforcement omits `records`/`p0_top` from its final pass predicate even though it computes their
equality. The audit checks both and they pass100/100. Resume paths trust existing status rows; future resumability
should verify manifest/identity/hash before reuse. No reuse mismatch is present in this completed run. No bug affecting
repaired lexical outputs was established.

## 3. Compact independent GPU reproduction

Audit selection is a fixed union of every changed repaired-TTLS row, every changed A2 row and all accepted AC rows
(22 unique), plus the first 2 canonical unchanged controls from an unrepresented dialogue:24 rows,14 dialogues.
This is outcome-stratified reproduction coverage, never an effectiveness estimate. It includes the new lexical phrase,
all prior harms, EOS recovery, the large capped repetition and unchanged controls.

The audit also checks clean and nonempty-zero free decoding and teacher-forced FFN/logit identity on **all 100 rows**,
with no optimizer. Full100 adapted episodes are not rerun. The independent FFN pre-hook records the native tensor
actually consumed at L16, rather than trusting a reconstructed recorder. Three selected queries have independent
complete-site gradient comparisons at initialization. No references enter any GPU path.

**Replay verification: PASS.** Job **58394** completed in **222.741 s** of measured work (Slurm allocation
4 min 5 s), exit0. All 100 clean baselines reproduced sealed B0 tokens, text and termination; all 100 nonempty-zero
free decodes, FFN inputs and logits were bitwise identical to clean. The old kernel independently changes logits on
100/100: maximum absolute difference .0683594–.25; consumed FFN chord .00564435–.0229671 even at zero z.

All **96 arm/row comparisons** (T1/T2/T4/T6 on 24 rows) reproduced tokens, text and termination exactly; active episodes
also matched all saved losses, gradient norms and final z arrays exactly (maximum differences 0). This includes 48
active CE/+P episodes, 12 active AC/+P episodes and 36 AC abstentions. All 6 accepted AC nonempty-mask zero controls
reproduced B0. Sixty native nonzero-consumption probes verified masked FFN displacement and norm bounds. Model
bytes stayed unchanged; no model gradients or site hooks leaked; per-episode state/reset checks passed.

The 3 independent full-site checks had exactly equal losses and z gradients (relative difference0, cosine≈1), and
nonzero CE gradient norms .126667, .0957068 and .194908. Pure-P at z=0 has value 0 but tiny gradient norms
1.29267e-7, 4.47789e-7 and 3.86536e-7, respectively (1.02e-6–4.68e-6 of the CE gradient). Adding such a term into a
shared BF16 backward graph can alter rounding of the larger gradient; it is not a sizable initial preservation force.

Reference-free raw outputs were sealed at **40d83ee8b1f294c8722de09d10e8747f752c033d**, pushed to the audit branch,
then scored by the separate independent post-seal script. Seal SHA256:
`d13518711e32beda88ede8c9ba6510f82bcb43092f10b693cf134be7683ef00e` (128 pinned files).
All 96 independently recomputed per-row count arrays and correction/corruption events match the original independent
scoring. No additional adapted full100 rerun is warranted.

Two failed audit-harness attempts are preserved:58392, missing `params` in the new independent loss wrapper;
58393, an incomplete independent gradient injection that omitted the FFN residual skip. Both stopped after the first
selected row; they do not invalidate Claude's run. The second-path repair was proven on a small BF16 CPU model with
exact loss/gradient equality before the final audit launch. No original experiment source or threshold was changed;
all outputs/logs remain separate, and the predeclared gradient comparison tolerances were not weakened.

## 4. Independent metric recomputation

The independent scorer reuses only the **earlier independent auditor's** canonical normalization/tagging, own
Wagner–Fischer alignment, count/taxonomy/transition primitives. It does not import Claude's original or repaired
aggregation, decision or bootstrap functions. Lexical classes, loading/merging, corpus sums, paired intervals and
first-token/EOS decomposition are independently implemented for R1R.

All 18 system arrays are scored from raw references and hypotheses. Corpus counts equal the sum of per-utterance
counts; PIER differences equal corrections−corruptions. There are zero published metric/transition mismatches.
Denominators are696 POIs,5771 mixed units,5048 ZH units,4054 baseline-correct CS-Mandarin units and4055 baseline-correct
outside-POI units. PIER excludes insertions while EN-WER includes them; equal POI net counts do not imply equal EN-WER.

| Method | PIER | MER | EN-WER | ZH-CER | POI corr/corrupt | Genuine / dialogues | New Mandarin corruptions |
|---|---:|---:|---:|---:|---|---|---:|
| B0 forced-ZH | 0.495690 | 0.256975 | 0.502874 | 0.219295 | 0/0 | 0 / 0 | 0 |
| AUTO | 0.408046 | 0.259054 | 0.415230 | 0.233558 | 63/2 | 37 / 3 | 72 |
| A2-CE | 0.454023 | 0.253682 | 0.462644 | 0.221078 | 31/2 | 15 / 1 | 26 |
| TTLS-CE repaired | 0.478448 | 0.251949 | 0.487069 | 0.215729 | 12/0 | 4 / 1 | 0 |
| TTLS-CE+P repaired | 0.487069 | 0.252989 | 0.495690 | 0.215729 | 6/0 | 0 / 0 | 0 |
| A2-CE+P | 0.455460 | 0.249523 | 0.464080 | 0.216125 | 30/2 | 15 / 1 | 3 |
| A2-AC | 0.489943 | 0.261133 | 0.497126 | 0.224842 | 4/0 | 1 / 1 | 30 |
| TTLS-AC repaired | 0.491379 | 0.260613 | 0.498563 | 0.224049 | 3/0 | 1 / 1 | 25 |
| A2-AC+P | 0.491379 | 0.261826 | 0.498563 | 0.225436 | 3/0 | 1 / 1 | 32 |
| TTLS-AC+P repaired | 0.491379 | 0.260613 | 0.498563 | 0.224049 | 3/0 | 1 / 1 | 25 |

T1 Mandarin retention=4054/4054; T2 likewise. A2=4028/4054; A2+P=4051/4054. T4/T6=4029/4054.
Outside-POI harm: T1/T2=0/4055, A2=26/4055, A2+P=3/4055, T4/T6=25/4055. These are observed development outcomes,
not a population risk bound. The active CE vector runs on 100 rows although only11/10 transcripts change; a
changed-output conditional denominator must not be called the number of physically edited episodes.

Mixed errors and S/D/I:

- B0:1483=339+1026+118.
- T1:1454=329+1007+118.
- T2:1460=334+1008+118.
- A2:1464=351+985+128.
- A2+P:1440=331+983+126.
- T4/T6/ACSUB:1504=334+1053+117.

## 5. Corrected versus original: exact decomposition

| Arm | R1→R1R token-changed rows | POI corrections R1→R1R | Genuine R1→R1R | Metric effect |
|---|---:|---|---|---|
| T1 |2|8→12|0→4|POI −4; mixed −4; ZH unchanged|
| T2 |0|6→6|0→0|none|
| T4 |0|3→3|1→1|none|
| T6 |0|3→3|1→1|none|

T1 changes on U0086_S0_185 and U0092_S0_101. The first recovers “go back to school” instead of the Mandarin
translation “回到学校”: four correlated POI units, **one phrase-level event**, one utterance/dialogue. The second
changes “更加”→“更为” with no canonical metric change. This is genuine but narrow lexical evidence, not four
independent recovery examples. It is sensitive to a native numerical-contract change and not a robustness result.

The repaired zero control=B0, so learned-T1 versuszero now legitimately measures the total learned-z intervention.
It has12 improvements/0POI corruptions, including4 genuine,3 word-boundary and5 deletion/boundary corrections, spread
across5 dialogues overall. Loss declines and the optimized activation is consumed. This is not merely a zero-control
rounding artifact. It is also not sufficiently broad evidence of CS-specific lexical superiority.

Four T1 EOS-recovery rows account for12/18 net ZH error reduction and14/19 net deletion reduction, but only2/12 POI
corrections (“super topic”). Thus Mandarin gains are largely termination-related; it would also be wrong to call all
English benefit EOS recovery. Three corrections are spacing repairs and a separate omitted-English-unit recovery
includes “it's”. The17 repaired baseline ZH units and18 total ZH-error reduction differ because insertions also count.

HistoricalzeroZ0H hasMER.241553 but changes6 token sequences and has a severe-length flag on a capped row. It remains
a preserved numerical-confound demonstration, not a comparator method to select or evidence against the valid
repaired T1 intervention.

## 6. Factorial, preservation and practical interpretation

**Matched supervision:** same AUTO CE teacher forA2/T1; same frozen clean/null acoustic candidate forAC arms;
same stable B0 distribution/set and λ1 for+P. No gold target enters inference. The actuator opportunities are still
unmatched: TTLS one L16 global vector vs194 LN tensors across the decoder; different LR/functional budgets; two
steps are not equal effective optimization. CE's pseudo-label path may already be on a different prefix than final
forced-ZH decoding. It does not solve the branch-entry problem.

**First-token limitation:** content step 0 is predicted at absolute query3. The frozen prefix mask excludes abs<4.
TTLS cannot change that decision in any arm; every saved TTLS first token equalsB0. A2 can. A2 changes first token on3
rows; A2+P on2; AUTO on12. All15 A2/A2+P genuine corrections and all 37 AUTO genuine corrections occur on these rows.
That is an important confound, not a proof that step 0 causally explains all their downstream gains: no matched
first-token ablation was run. Comparisons cannot isolate “representations are better/worse than parameters.”

**Preservation:** A2+P retains30/31 POI corrections and all 15 genuine units, reducing new Mandarin corruptions26→3.
T1→T2 loses6 corrections, including all 4 genuine units, and avoids no observed additional Mandarin harm. T4/T6 are
token-identical. KL-to-the-B0 teacher path cannot guarantee the steered free path remains safe.

A2+P is the more useful development trade-off, but it is not broadly validated: its15 lexical corrections all come
from U0021_S0_513;20 of23 avoided Mandarin corruptions come from U0023_S0_664. It is a prospective candidate,
not an established general superiority claim. TTLS's0 observed Mandarin corruptions accompanies much lower English
benefit; no dose/correction-matched Pareto comparison establishes a better trade-off.

**Acoustic objective:**6/100 accepted targets,94% abstention. TTLS-AC/T6 have the same normalized token-unit sequences
and all per-utterance counts asACSUB100/100, raw normalized strings99/100 (punctuation differs on U0012_S0_103),
exact tokens98/100. No canonical word-recognition gain comes from optimizingz over simplecandidate substitution.
Only one genuine correction (“cooking”) is shared byboth, with 25 new Mandarin corruptions. U1003_S0_63 alone
contributes23:92 baselinecontent tokens→64, +27 deletions. Its strict-prefix premature-EOS counter and half-length
severe counter are zero because the prefix changed and remaining length exceeds half. A zero severe flag is not proof of
no termination damage. The candidate's reference-word-prefix match somewhere in a transcript is not aligned target
precision, particularly for short “I”, “c” or “or”.

**AUTO and runtime:** ordinaryAUTO PIER.408046 is better thanT1.478448 andA2+P.455460, at worseMandarin error rate.
TTLS offers no demonstrated all-metric dominance. T1 adaptation9.85s/100, decoding55.3s/100: tiny adaptation cost
does not cure limited utility; do not claim unrestricted activation optimization is expensive merely from theory.
Audit timings include extra verification and are not deployment latency.

## 7. Dialogue-level paired uncertainty

Independent20-dialogue paired bootstrap, 2000 draws, seed 240924. Resample dialogue blocks and recompute corpus
micro-ratios from summed counts. No query/word independence assumption. All draws have positive denominators.
Intervals are descriptive, unadjusted across an exploratory family and not independent-confirmation/significance
claims; they do not account for repeated historical selection on the same dialogues.

| Contrast (method minus comparator) | ΔPIER [95%] | ΔMER [95%] | ΔZH-CER [95%] |
|---|---|---|---|
| T1-B0 | -0.017241 [-0.033616, -0.004081] | -0.005025 [-0.008797, -0.001407] | -0.003566 [-0.007221, -0.000260] |
| T1-B2 | +0.024425 [-0.019764, +0.077410] | -0.001733 [-0.013128, +0.007384] | -0.005349 [-0.017388, +0.002255] |
| T2-T2A | +0.031609 [-0.007353, +0.082817] | +0.003466 [-0.001287, +0.010616] | -0.000396 [-0.005204, +0.004445] |
| T1-B1 | +0.070402 [+0.000000, +0.148203] | -0.007104 [-0.017974, +0.004317] | -0.017829 [-0.037006, -0.003846] |
| T1-T1_R1 | -0.005747 [-0.018519, +0.000000] | -0.000693 [-0.002127, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| T2-T1 | +0.008621 [+0.000000, +0.024062] | +0.001040 [+0.000000, +0.002573] | +0.000000 [+0.000000, +0.000000] |
| T4-T3 | +0.001437 [+0.000000, +0.004599] | -0.000520 [-0.002482, +0.000730] | -0.000792 [-0.002941, +0.000574] |
| T6-T5 | +0.000000 [+0.000000, +0.000000] | -0.001213 [-0.003723, +0.000000] | -0.001387 [-0.004245, +0.000000] |
| T4-ACSUB | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |

Positive aggregate T1–B0 intervals support an observed small gain; they do not establish broad lexical benefit or
superiority toA2. The four lexical units occur together in one dialogue and disappear together in cluster draws
omitting it. Zero-corruption resampling can give degenerate intervals and cannot certify zero future risk.

## 8. Strongest reviewer objections and recommendation

1. **Insufficient independent lexical events:** four tokens of one phrase are not four independent examples or four
   dialogues. Other CE gains largely concern word boundaries and termination. No genuine same-language wrong-word
   repair is observed for this TTLS configuration.
2. **Actuator comparison confounded:** global-vector/local-layer restriction, first-token exclusion and functional
   budgets prevent a clean adaptation-space conclusion even though pseudo-supervision is matched.
3. **Preservation does not add TTLS utility:** it removes the only new genuine phrase while A2+P retains its own
   lexical gains and avoids most observed harm. A fair comparison needs more than lower raw corruption counts.
4. **Acoustic method unnecessary/unsafe:** sparse accepted evidence; direct substitution reproduces canonical results;
   one large continuation deletion event dominates harms. Local candidate rank is not free-sequence correctness.
5. **Numerical sensitivity remains:** repairing identity changes one whole phrase. This is a valid kernel revision,
   not evidence of a stable learned linguistic axis. No matched lexical-gradient/D3 alignment at these exact queries
   is sealed, and no semantic interpretation follows from loss reduction or hidden-state cosine.
6. **Repeated development exposure:**100 rows/20 reused dialogues, outcome-informed repair and diagnostic design,
   concentrated A2 andTTLS lexical gains. Neither method is independently confirmed here.
7. **Mechanistic reporting needs precision:** zero forwardKL does not guarantee an exactlyzero native backward;
   zero strict-prefix EOS flags do not exclude damaging shorter continuations. These do not invalidate the repaired
   metrics, but they constrain interpretation.

Numerical validity **PASS**; independent corpus reproduction **PASS**; compact method reproduction **PASS**;
evidence that this TTLS is promising **NO**. The independently recomputed frozen verdict is
`TTLS_R1R_VALID_BUT_INSUFFICIENT`, consistent with the report. Audit verdict: **`AUDIT_PASS_TTLS_INSUFFICIENT`**.

**One clear recommendation: prioritize A2+P instead.** Close this TTLS version rather than launch incremental LR,
step, layer, mask or direction searches. A fundamental opportunity-matched representation-adaptation redesign would
be a different scientific hypothesis and require new human authorization, not a continuation justified by this run.
A2+P also needs a separately frozen prospective, exposure-safe protocol and explicit approval. No next experiment
is started by this audit.

## 9. Audit artifacts and tests

- `experiments/ttls_r1r_independent_analysis.py`: independent repaired loading/scoring/lexical split/cluster bootstrap.
- `experiments/ttls_r1r_independent_replay.py`: zero-only100 census, compact24 adaptation replay, native FFN probes,
  independent complete-site gradients and numerical-P probes; no reference loader.
- `experiments/ttls_r1r_independent_replay_score.py`: verify sealed replay hashes and independently rescore all 96
  method/case outputs; recompute the frozen necessary lexical-breadth gate without importing primary decision logic.
- `tests/test_ttls_r1r_independent_audit.py`: original failure, actual nonempty-zero identity, analytic/directional/native
  gradients, full residual-path gradient, corpus count identity and history preservation.
- `slurm/ttls_r1r_independent_replay.sbatch`: bounded H100 MIG replay only.
- `results/inference_cf/ttls_r1r_independent_audit/`: exact IDs/config/source pins, independent metrics/rawevents,
  decomposition, manifests, preserved failed attempts, completed rawreplay and outputseal.

CPU tests: 51 focused independent/repaired/original audit+TTLS tests and 47 historicalMER/PIER/retention/canonicalmetrics/
episodicTTA tests, **98 passed**. Existing scalar-loss warning and serial joblib fallback from a full `/dev/shm` were
observed; neither detaches training. No historical source was patched. Activejobs and branch equality are verified
at final publication.

### Reproduction commands and environment

Run from the audit worktree using `/home/tungnx/miniconda3/envs/acl1/bin/python`. Set
`LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib` and
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (as the audited Slurm script does).
The final CPU check initially omitted the conda library path and failed to import SciPy because the system
libstdc++ lacks `CXXABI_1.3.15`; this environment failure is preserved separately. Restoring the execution environment
requires no source or test change.

```bash
python experiments/ttls_r1r_independent_analysis.py
python experiments/ttls_r1r_independent_replay_score.py
python -m pytest -q tests/test_ttls_r1r_independent_audit.py tests/test_ttls_r1r.py \
  tests/test_ttls_r1.py tests/test_ttls_independent_audit.py tests/test_mer.py \
  tests/test_pier.py tests/test_retention.py tests/test_canonical_metrics.py \
  tests/test_inference_cf_p2tta0.py
```

The bounded GPU script and its source/model manifest are archived; the completed independent replay is already
sealed. Reproduction commands are provenance, not authorization for another job or method search.
