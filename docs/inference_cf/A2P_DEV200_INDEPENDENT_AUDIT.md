# A2P-DEV200 — independent scientific audit

**Technical verdict: `A2P_DEV200_AUDIT_PASS`.**

**Scientific verdict: `A2P_DEV200_NOT_SUPPORTED`.**

**Recommendation: Stop the A2+P direction due to insufficient evidence of incremental preservation benefit.**
This recommendation concerns the exact frozen T2A/A2-CE+P method. It does not invalidate historical A2, prove that
all preservation objectives fail, or authorize a replacement objective, tuning, another development round or confirmation.

The implementation is reproducible. Independent scoring agrees with the published corpus metrics and transition
counts; a separately frozen 24-case GPU replay matches A2 and A2+P exactly. However, NEW200 contains **zero Mandarin
corruptions prevented by preservation**, the same nine baseline-correct Mandarin units are damaged by both methods,
and A2+P adds ten total Mandarin errors and eight mixed errors relative to A2. Its sole additional POI correction is
`am` → `I'm`. The fixed100 protection signal is dominated by one utterance; pooling fixed100 with NEW200 obscures
that failure to reproduce the pattern on additional development utterances.

## 1. Executed revision, data and provenance

Audit worktree: `/home/tungnx/cs-asr-steer-a2p-dev200-audit`; branch `audit/a2p-dev200-independent`.
The original `/home/tungnx/cs-asr-steer-inf` worktree and `feature/a2p-dev200` experiment remain unchanged.
Remote is `https://github.com/XuanTung2k3/cs-asr-steer.git`.

| Record | Verified value |
|---|---|
| Completed experiment revision | `4bbc0f83bd8f440a196c62a64a65025b9fc07420` |
| Config freeze | `25190862e5bf3896203fbc94f7a1127a53341003` |
| Spec/code/plan freeze | `6a642301fed24d52d08b329abcd3bc02702eed92` |
| Original execution manifest commit | `8d94411f7180a8897723d6c1a77bc319de205967` |
| Original output-seal commit | `0dbd52e6be53f3cb64472831bcd528fbe37ab332` |
| Independent analysis/replay source freeze | `95c7729a` |
| Independent replay manifest commit | `d80edd85a54624c9f30bbf0d2431dc5bd560dbcb` |
| Independent replay output-seal commit | `0bd5f16e` |
| Original job | Slurm `58397`, 200 NEW200 episodes plus 16 fixed100 regression cases, 8m26s |
| Independent replay | Slurm `58400`, 24 NEW200 episodes, COMPLETED, exit 0, 1m29s wall time |

Original config SHA256: `235deb7c5fdb274179048878bd580228f838841c2bdb9aa023be926fbfb5dd36`.
Original plan hash: `sha256:c39d964c9ca460ef0debd905dce11607a0b52f23af1e0d18a0a06b9970c4bd75`.
Original manifest hash: `sha256:8338c982df0ed5d2c40c5c81bd37102e9eae23c15b906daf242b05b396e09ff0`.
Original seal hash: `sha256:aaccd4dda88ac83bc8adc3cc1c075f2c02f15930f0fa80503a44eab925b64a87`.
All 220 sealed files, manifest source pins, and inherited PATH5 and TTLS-R1 sealed files verify independently.
The seal commit is an ancestor of, and predates, the evaluation/report commit. Runner inputs and imports contain
no references. This supports the documented seal-before-evaluation order; Git chronology is not a complete
forensic record of every historical filesystem read.

Population manifest: `docs/inference_cf/P2_PATH5_PANEL.json`, SHA256
`4386b1fcb272e1892c8d3766a4c305e4ae8700d0b5cf63f74bbc67a7b0adc5a2`.
The independent analysis checks canonical order, uniqueness and NEW200 = FULL300 minus FIXED100 exactly:

| Partition | Utterances | Dialogues | Membership SHA256 |
|---|---:|---:|---|
| FIXED100 | 100 | 20 | `15c0c82e1a4285f85a9ac51d395d4e26549b34a218f48ddbde75a3d91657e94d` |
| NEW200 | 200 | 20, ten utterances each | `e23a8ce549820a8aa4287eceb3bce71ea2d8aae1af52b9a50bab6a228a9b2b40` |
| FULL300 | 300 | 20 | `bc8062390705cf133cbd42813d01b65f7ff3d5db836c5b94d9fc602275db6c5c` |

Exact IDs and dialogues are stored in the unchanged panel and independent per-row results. Every audio SHA is
verified. All FULL300 IDs appear in the hash-validated development exposure registry. References are filtered by
these IDs from `role_D-dev-select.parquet`; reference-file SHA256 is
`25a45f28c0e192c6b2d71bdd0eccbfe354ac1a4e0019cddd78168d86423d2109`.
No confirmation, test, calibration, transfer or new-role data enters this audit. NEW200 was already exposed in
earlier development work, and uses the same 20 dialogues. It is additional utterance-level evidence, not independent
statistical confirmation or evidence of generalization to new speakers/dialogues.

### Model and decoding

Local frozen model/tokenizer: `/mnt/data/tungnx/whisper-large-v3`; all eleven file hashes match the original
manifest, PATH5 and TTLS-R1. The replay verifies every local file before model loading.
`model.safetensors`: `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`.
`tokenizer.json`: `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca`.
All remaining tokenizer/config/preprocessing hashes are in `independent_metrics.json` and the original manifest.
An upstream Hugging Face revision is not recorded; reproducibility is by local snapshot bytes, not a claimed HF commit.

Python 3.11.9, Torch 2.10.0+cu128, Transformers 4.57.6, NumPy 1.26.4, eager attention, BF16 model forward,
greedy forced-ZH decoding, prompt `[50258,50260,50360,50364]`, no timestamps, cap 200, seed 240924.
The source-compatible preprocessing, suppression and beginning-suppression lists are pinned in the plan.
Content arrays omit EOS; termination is checked separately.

## 2. Independent implementation audit

Reviewed AGENTS, METHOD_CONTRACT, exposure ledger/registry, A2P spec/config/runner, original TTLS-R1 T2A,
`ttls.py`, `episodic_tta.py`, `soft_auto_tta.py`, and TTA1, A2-MECH0 and PATH5 history. The exploratory TTA
ticket is distinct from the core-v6 steering proposal. No historical scientific contracts or kernels were changed.

| Invariant | Finding |
|---|---|
| Exact inherited T2A | Same A2 AUTO-consistency CE plus lambda=1 preservation KL; no objective replacement |
| Teacher/student | Sealed theta0 AUTO content IDs, forced-ZH student; valid target mask fixed before adaptation |
| Trainables | Exactly 194 decoder LayerNorm weight/bias tensors, 248,320 scalars |
| Optimizer | Fresh AdamW, lr .001, wd 0, betas .9/.999, eps 1e-8; amsgrad/foreach/fused/maximize false |
| Update schedule | Exactly two optimizer updates and three loss evaluations; no tuning |
| Precision | FP32 master parameters cast to BF16 through differentiable functional_call |
| Preservation target | Frozen theta0 allowed-vocabulary distribution on the B0 prefix, not AUTO or adapted targets |
| Preservation direction | Forward KL(theta0 || adapted), averaged over stable positions |
| Stable set | Valid B0 positions with p0(B0 token) >= .5, excluding every position in M |
| M construction | Frozen UTF-8/embedded-token clean-EN-vs-clean-ZH candidate logic; M precedes acoustic acceptance |
| Acoustic acceptance | Both prompt branches require clean-vs-30s-zero-audio evidence >= log(10), inherited null floor |
| Episodic state | Fresh FP32 masters and optimizer per utterance/arm; resident LN bytes restored exactly |
| Non-LN parameters | Frozen/no gradients; original raw hashes match, replay all-parameter byte hash unchanged |
| Free decoding | Effective adapted BF16 LN state is materialized and genuinely used in the final cached greedy decode |
| Gold information | No reference text, gold language labels, gold location, target tokens or gold timings in adaptation |

**M is not just the first accepted acoustic candidate.** Even a rejected acoustic candidate belongs to M and is
excluded from S. The independent implementation specifically tests this distinction. Stable position zero is
allowed for LayerNorm TTA; the first-content-token restriction of the old TTLS activation hook does not apply.
Content CE and S exclude the terminal EOS query. Adaptation can nevertheless change EOS decisions indirectly.
The objective is a soft penalty, not a hard free-decoding retention constraint or a language-local parameter edit.

Initial preservation values are exactly 0 for all 200 records. Float32 KL backward at numerical equality has a
small rounding residual, so the A2+P initial gradient need not equal A2's bitwise. The independently implemented
same-numerics gradient matches the production A2+P gradient exactly; this inherited behavior was not silently repaired.

The original job verified live B0 and AUTO for every NEW200 row (tokens/text/termination all equal in saved records).
A2 is correctly reused from audited PATH5, not an incompatible older B0 path. Production's all200 A2 check is
step-0 CE identity, not an all200 replay of final A2. The independent replay strengthens this with exact final
A2 states and outputs on 24 targeted cases. Separately, all sixteen saved fixed100 regression gates pass,
including T2A histories and historical A2 FP32-archive reproduction.

No material implementation or baseline mismatch was found. Original results remain valid and unchanged.
Minor reproducibility limitations: B3's original full FP32 master/optimizer tensors were not archived (norms and
effective BF16 hashes were); the upstream model revision is absent; the resume shortcut trusts existing `ok` rows
and would need stronger binding for a future run. No evidence these limitations changed this completed run exists.

## 3. Independent evaluator and exact recognition totals

`experiments/a2p_dev200_independent_analysis.py` reuses only the earlier **independent auditor's** normalization,
tagging and dynamic-programming primitives, not Claude's aggregation, metric-reporting, bootstrap or decision code.
Wagner–Fischer alignment uses verified diagonal/deletion/insertion tie order. Han characters and English units follow
canonical normalization. Corpus metrics are ratios of summed counts, not means of utterance rates.
The follow-up independent review reconstructs outside-region projections, transitions, lexical categories and harm
identities, then compares them to the published JSON. Every corpus count equals its per-row sum, and all published
PIER/MER/EN-WER/ZH-CER, S/D/I, correction/corruption, ZH, outside-harm and sequence-event totals agree.

### NEW200 (primary)

| System | PIER | MER | EN-WER | ZH-CER | POI errors | Mixed errors | ZH errors | S / D / I |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0 forced-ZH | .463740 | .268758 | .465649 | .233016 | 729 | 2973 | 2202 | 748 / 1986 / 239 |
| AUTO | **.383588** | .260803 | **.388041** | .236720 | **603** | 2885 | 2237 | 652 / 2013 / 220 |
| A2 | .442748 | **.240644** | .446565 | **.203280** | 696 | **2662** | **1921** | 728 / 1694 / 240 |
| A2+P | .442112 | .241367 | .445293 | .204339 | 695 | 2670 | 1931 | 727 / 1699 / 244 |

Denominators: 1,572 POI/EN units, 11,062 mixed reference units, 9,450 ZH units.
POI errors exclude insertions, while EN-WER includes EN insertions; the columns must not be treated as interchangeable.

| Changes relative to B0 | AUTO | A2 | A2+P |
|---|---:|---:|---:|
| POI corrections / corruptions | 129 / 3 | 36 / 3 | 37 / 3 |
| Genuine lexical units (strict boundary exclusion) | 115 | 22 | 23 |
| Wrong-language / same-language lexical units | 112 / 3 | 21 / 1 | 22 / 1 |
| Boundary repairs / deletion-or-boundary recovery | 1 / 13 | 1 / 13 | 1 / 13 |
| Genuine phrase events / utterances / dialogues | 26 / 8 / 7 | 6 / 3 / 3 | 6 / 3 / 3 |
| Newly incorrect baseline-correct ZH units | 60 | 9 | 9 |
| Previously incorrect ZH units repaired | 2 | 288 | 283 |
| Changed outputs / EOS-prefix recoveries | 24 / 0 | 30 / 8 | 30 / 8 |

For A2 and A2+P, matrix-ZH retention is 7,474/7,483 = .998797; English retention 840/843 = .996441;
baseline-correct outside-POI harm is 9/7,485 = .001202. Both methods corrupt the **same nine reference indices**:
U0023_S0_469 (1, CSD0012), U0040_S0_149 (3, CSD0020), U1001_S0_388 (3, CSD0501), U1027_S0_3 (2, CSD0514).
There are no new severe truncations or caps; one pre-existing capped repetitive output remains in every system.
Absence of new severe truncation does not make that existing failure harmless.

**Direct A2+P versus A2:** one extra POI correction, zero POI corruptions, zero additional wrong-language corrections,
five A2-correct ZH units lost and none repaired; outside-POI harm 5/7,764. Total ZH errors increase by **10**, not five:
five are lost reference-unit matches in U0040_S0_149, and five are extra ZH insertion errors in U0091_S0_104.
Overall S/D/I changes are −1/+5/+4, yielding eight extra mixed errors. Equal new-ZH counts versus B0 do not capture
these lost A2 repairs or insertion damage.

### Fixed100 and FULL300 (secondary)

| Population | System | PIER | MER | ZH-CER | POI / mixed / ZH errors | New ZH errors vs B0 |
|---|---|---:|---:|---:|---|---:|
| fixed100 | A2 | .454023 | .253682 | .221078 | 316 / 1464 / 1116 | 26 |
| fixed100 | A2+P | .455460 | .249523 | .216125 | 317 / 1440 / 1091 | 3 |
| NEW200 | A2 | .442748 | .240644 | .203280 | 696 / 2662 / 1921 | 9 |
| NEW200 | A2+P | .442112 | .241367 | .204339 | 695 / 2670 / 1931 | 9 |
| FULL300 | A2 | .446208 | .245114 | .209477 | 1012 / 4126 / 3037 | 35 |
| FULL300 | A2+P | .446208 | .244163 | .208443 | 1012 / 4110 / 3022 | 12 |

FULL300 simply combines old fixed100 protection with NEW200 failure; it is not a new independent evaluation.

## 4. Compact independent GPU replay

Selection was frozen before replay: union all five A2+P-vs-A2 changes, all three genuine-recovery utterances, all
four Mandarin-harm rows, and all eight EOS-recovery rows; fill with the first canonical unchanged control in each
unrepresented dialogue, then canonical unchanged controls, to reach 24. It spans **all twenty dialogues** and includes
entirely correct B0 controls and the largest/capped changes. Categories are present. This outcome-stratified audit
subset is not a representative estimate of method effectiveness.

NEW200 indices: `[0,1,3,11,24,31,40,49,53,58,61,70,80,94,100,110,122,139,140,150,163,179,182,190]`.
Exact IDs, selection rationale and source/config hashes are in `replay_selection.json` and `replay_manifest.json`.
Replay manifest hash: `sha256:17a9f63d4e60afcd168d0bdc746b74f8b0cfbb23806fed03536966ea2d4b81c9`.

The separate reference-free harness independently implements M/S, allowed-vocabulary NLL, theta0 forward KL,
and candidate acoustic acceptance; it reuses the frozen optimizer/model primitives rather than rewriting Whisper.
It checks independent versus original initial losses and every LN gradient, then runs two updates with its own
loss calculation. It separately replays historical A2 without preservation.

| Verification | Result |
|---|---|
| Original model/tokenizer/preprocessing bytes | All eleven model-file pins pass before loading |
| B0 generated tokens/text/termination | 24/24 exact |
| M, acceptance, first candidate, S and p0 | 24/24 exact |
| Independent loss and initial LN gradients vs frozen B3 | 24/24 bitwise exact |
| Three B3 losses/CE/P parts, two gradient norms, master/effective delta norms | 24/24 exact |
| Final effective B3 LN hash, tokens/text/termination | 24/24 exact |
| Historical A2 losses, final effective LN hash and free decode | 24/24 exact |
| Fresh optimizer, theta0 masters, reset and hook isolation | Every replay episode passes |
| All resident pretrained parameters at end | Exact initial byte hash; no gradients |
| Independent post-seal replay scoring | Counts and ZH harm/repair identities exact on 24/24 |

No BF16 discrepancy or material error occurred. Full200 rerun was unnecessary. The harness also archives its own
FP32 master and AdamW-state hashes; these are new audit evidence, not falsely claimed comparisons to unavailable
original B3 optimizer archives. Job body took 65.14s; Slurm total 89s. Raw outputs were committed/pushed and sealed
before replay reference rescoring. Replay seal: `sha256:db3b89dd84be01d2c9043453168d001f43bc12aee6a81d0746d054067c3e971b`.

CPU verification: 73 relevant existing tests passed; all three new focused independent-audit tests passed.
The first focused run incorrectly compared elapsed adaptation time and failed only on that field; it was fixed
before GPU source freeze. No method code changed. Warnings about serial joblib and scalar logging are nonfatal.

## 5. Preservation, lexical recovery and mechanism

NEW200 does **not** reproduce the fixed100 preservation pattern. A2+P retains all 36 A2 POI corrections and all
22 genuine lexical units, but prevents zero new Mandarin corruptions, repairs no new Mandarin units relative to A2,
and loses five A2 repairs. It damages no additional A2-correct English units. Five transcripts differ, of which only
one improves mixed errors and two worsen them; two changes are score ties.

The extra correction in U1027_S0_3 is an English contraction (`am` → `I'm`) relative to A2. Comparing only against
B0 labels that POI unit a wrong-language recovery, because B0 used Chinese there. That must not be attributed to
preservation as a new wrong-language correction: A2 already switched to English. Correct English words within the
same repaired phrase are correlated, not 22 or 23 independent demonstrations of lexical power.

The six genuine events occupy three utterances/dialogues: U0029_S0_1085, U0040_S0_454 and U1027_S0_3.
They mostly recover English-language renderings of predominantly English speech, not broad localized embedded-English
repair. U1027 is also the sole changed first-content-token case. The fixed100 fifteen genuine lexical units occur
in two phrase runs of one utterance/dialogue, U0021_S0_513. These narrow trajectories limit code-switching claims.

The original 180-position steering panel is not this TTA population. This audit uses complete frozen fixed100 and
NEW200 utterances and evaluates free decoding; teacher-forced logits are not substituted for transcript evidence.

Teacher-forced preservation KL is not a guarantee of output preservation. Across NEW200, final stable KL has mean
.03290, median .02373 and maximum .57842 (the repetitive capped row). Every original S is nonempty (mean 40.78),
while only 27/200 rows have nonempty M. The same clean-prefix target need not constrain an adapted divergent prefix.
Observed transcripts establish that preservation changed particular trajectories; they do not identify a unique
branch mediator or show localized, language-specific LN parameters. No extra ablation or objective search was run.

Independent token-array inspection does support a limited trajectory observation. On fixed100 U0023_S0_664,
preservation moves the first B0 divergence from content position 0 to position 10; three other A2-divergent rows
become exact B0. On NEW200, it does not change any first-content-token decision. It changes the continuation action
at B0's EOS site 45 on U0040_S0_149, and moves the first divergence **earlier**, from 51 to 42, on U0091_S0_104.
Both changed continuations worsen ZH errors. This is consistent with a soft global constraint sometimes avoiding
a bad branch and sometimes perturbing a useful continuation; it is not broad free-decoding protection.

EOS/deletion effects dominate the underlying A2 benefit. Eight exact-prefix EOS-recovery rows account for 274 of
A2's 281 net ZH-error improvement, and 269 of A2+P's 271 (97.5% and 99.3%). Their net deletion reductions are 289
and 284; overall deletion reductions are 292 and 287. Eleven POI corrections occur on these EOS-recovery rows;
thirteen POI recoveries are deletion/boundary-classified overall. The genuine lexical subset is separate and narrow.
P neither adds nor removes an EOS-recovery case; it loses five errors worth of continuation fidelity on one existing
recovery. No explicit EOS loss was optimized, and causal claims about why global LN changes recover EOS require
the historical mechanism evidence rather than these aggregate numbers alone.

## 6. Breadth, concentration and uncertainty

For NEW200 mixed errors, A2+P versus B0 improves 12 utterances, worsens 5 and ties 183; by dialogue, 9 improve,
3 worsen and 8 tie. A2 improves 13/4/183 utterances and 10/2/8 dialogues. **Incremental P versus A2** improves
1 utterance/dialogue, worsens 2 utterances/dialogues and ties 197 utterances/17 dialogues. For new-ZH protection,
all 200 utterances and all 20 dialogues tie exactly: every d_u is zero.

The largest NEW200 A2+P mixed-error rescue is U1003_S0_119: 123/309 positive rescued errors (39.8%). The top
three utterances account for 82.2%; top three dialogues 83.0%. For ZH rescue, the largest share is 44.6%, top three
utterances 88.0%, top three dialogues 88.4%. These are positive-rescue shares; net harms are also retained.
Removing the most influential NEW200 dialogue still leaves **180 net mixed and 148 net ZH errors rescued versus B0**.
Across all twenty leave-one-dialogue-out panels, mixed rescue is 180–304 and ZH rescue 148–273. Thus underlying
A2-type aggregate benefit is not erased by a single dialogue, although it is concentrated in EOS recovery.

The added **preservation** benefit is different. Fixed100 prevents 23 new ZH corruptions, of which 20 (87.0%) come
from U0023_S0_664/CSD0012 and three from U0101_S0_190. Removing CSD0012 from fixed100 leaves three prevented
corruptions, with only .000706 MER improvement. NEW200 prevention is zero under every dialogue exclusion; its
P-vs-A2 MER difference remains adverse under every exclusion (+.000284 to +.000936).
FULL300 shows 15 fewer total ZH and 16 fewer mixed errors for P, but **removing CSD0012 reverses both advantages**:
five more ZH errors and four more mixed errors, ΔZH-CER +.000352 and ΔMER +.000244. The pooled advantage therefore
does not supply the requested robust preservation evidence. Protection share is undefined where no errors are
prevented; zero prevention is not reported as a well-distributed benefit.

Independent paired bootstrap resamples all twenty dialogues with replacement, keeping every paired utterance and
recomputing corpus ratios. Seed 240924, 2,000 draws, percentile 95% intervals; descriptive, no selection-corrected
significance claim or utterance-independence assumption.

| NEW200 contrast | ΔPIER [95% interval] | ΔMER [95% interval] | ΔZH-CER [95% interval] |
|---|---|---|---|
| A2+P − A2 | −.000636 [−.001960, 0] | +.000723 [−.000351, +.002137] | +.001058 [0, +.002695] |
| A2+P − B0 | −.021628 [−.0419, −.0054] | −.027391 [−.0540, −.0050] | −.028677 [−.0598, −.0034] |
| A2+P − AUTO | +.058524 [+.0006, +.1238] | −.019436 [−.0502, +.0085] | −.032381 [−.0634, −.0068] |

Exact intervals, all paired contrasts, per-row and per-dialogue totals, and every dialogue exclusion are stored in
`independent_metrics.json`. The [0,0] interval for prevented new ZH corruptions is an empirical degeneracy, not proof
of population equivalence or a universal absence of preservation benefit. Only nine A2 baseline-correct ZH harm
opportunities exist on NEW200; power to establish a general prevention effect is limited. Repeated development
selection and shared dialogues also limit all confidence intervals.

## 7. AUTO and practical value

A2+P has lower NEW200 mixed and ZH errors than AUTO, but **92 more POI errors** (695 vs 603), PIER .4421 vs .3836,
and EN-WER .4453 vs .3880. Its English limitation is substantial. The observed MER advantage over AUTO has a
dialogue-bootstrap interval spanning zero. It is a language-accuracy trade-off, not uniform superiority.

Preservation requires frozen B0 targets, clean/null dual-prompt candidate passes and extra B0-prefix KL forwards
in addition to two updates and final decode. Original measured NEW200 time is 439s (about 2.20s/utterance) including
integrity/instrumentation; adaptation itself totals 41.26s and B3 decoding 112.02s. This is not a clean matched
latency benchmark against AUTO or A2, and no speedup/overhead ratio is inferred. Given the absent incremental
Mandarin protection, those extra components have no demonstrated NEW200 payoff.

## 8. Reviewer assessment and decision

**Strongest evidence for continuation:** exact reproducibility, clean role separation, unchanged matched supervision,
real free decoding, retained A2 lexical corrections, and genuine fixed100 prevention. The observed A2/A2+P Mandarin
advantage over B0/AUTO remains after removing one NEW200 dialogue.

**Strongest evidence against this A2+P method:** no prevented harm on NEW200, ten added ZH errors relative to A2,
no additional wrong-language correction, narrow lexical events, EOS-dominated aggregate gains inherited from A2,
87% of fixed100 protection in one utterance, reversal of pooled incremental benefit after removing that dialogue,
and materially worse English recognition than ordinary AUTO. Standard KL regularization alone does not establish
a novel, broad code-switching preservation mechanism or justify the implementation complexity.

Answers to the skeptical review questions:

1. Additional NEW200 utterances do not reproduce the original preservation pattern.
2. New Mandarin corruptions are not reduced: same nine units and four dialogues, zero prevention.
3. A2's genuine corrections are retained across three dialogues, but P adds no wrong-language recovery.
4. Underlying A2 gains survive removing a dialogue; the pooled **incremental P** advantage does not.
5. A2+P trades worse English/PIER for lower ZH error versus AUTO; P does not improve Mandarin over matched A2.
6. EOS/deletion recovery explains almost all net ZH benefit; this is predominantly the known A2 effect.
7. This is useful negative development evidence, not an independent population confirmation.
8. The frozen preservation method is not sufficiently broad/effective to warrant prospective confirmation or a
   paper-level effectiveness claim from these results.

The independent frozen-rule calculation gives N2=9, N3=9, D=0, activating NOT_SUPPORTED before other gates.
Correction retention is 100%, but cannot override zero protection. Opportunity, magnitude and breadth gates fail;
every leave-one-dialogue-out protection count is zero. Technical passing and scientific success are separate.

**Final recommendation: Stop the A2+P direction due to insufficient evidence.** No further experiment was started.
Any fundamentally different preservation hypothesis needs a new human scientific decision; this audit does not
authorize confirmation, parameter/objective sweeps or alterations to A2.

## 9. Supporting artifacts and reproduction

Under `results/inference_cf/a2p_dev200_independent_audit/`: `independent_metrics.json`, `review_crosschecks.json`,
`replay_selection.json`, `replay_manifest.json`, `replay_output_seal.json`, `replay/rows/*.json`, `replay/summary.json`,
job log and focused CPU test log. They preserve the original outputs and store only new audit evidence.

Independent commands (project interpreter, conda library path, single CPU threads):

```bash
python experiments/a2p_dev200_independent_analysis.py
python experiments/a2p_dev200_independent_review.py
python -m pytest -q tests/test_a2p_dev200_independent_audit.py tests/test_a2p_dev200.py \
  tests/test_ttls_r1.py tests/test_ttls_r1r.py tests/test_ttls_independent_audit.py \
  tests/test_ttls_r1r_independent_audit.py tests/test_inference_cf_p2tta0.py
```

The replay entry is `experiments/a2p_dev200_independent_replay.py`; frozen Slurm file is
`slurm/a2p_dev200_independent_replay.sbatch`. It refuses to overwrite an existing audit attempt.
No full200 rerun, new hyperparameter, alternate objective, localization search or subsequent experiment occurred.
