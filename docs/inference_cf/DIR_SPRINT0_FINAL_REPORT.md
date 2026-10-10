# DIR-SPRINT0 final report — three new steering-direction families at L16 DG-02

**Terminal label: `DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE`.** The evaluator's first-matching label under the frozen
precedence is confirmed by the independent FULL audit (`DIR_SPRINT0_AUDIT_FULL: PASS`, 26/26).

**Family results.**
- **D3 acoustic-prior contrast:** valid and tested, lexical power insufficient (2 corrections).
- **D4 transcription-fidelity contrast:** valid and tested, lexical power insufficient (0 corrections).
- **D5 phonetic-concept steering:** valid and tested, lexical power insufficient (0 corrections).
- **Gated variants:** none of the three R2-gated variants corrected anything.
- **Mechanistic effect:** no family shows one. Gap closure beyond matched random is at most about 0.01 nat.

**Recommendation.**
- **Close DIR-SPRINT0**, and with it the steering-direction-discovery branch.
- No direction is worth a full-decoding study.
- Any return to the separately frozen A2 confirmation needs explicit human authorization.

**Scope.** Local, single-query development evidence on 240 new utterances from the same 20 previously used dialogues.
This is not confirmation, not a full-ASR claim and not test-role evidence.

## 1. Frozen design and Git hashes

| Item | Value |
|---|---|
| Starting HEAD (D5 provider ready) | `930bca0d` |
| Design freeze | `c54d5027`: spec, design (adversarial review), firewall, compute, population, config `sha256:b264460b…`, D5 concept table `sha256:cbd856f3…`, threshold engineering, provider apparatus reference, freeze index `sha256:54da35aa…` |
| Implementation | `76adcfba`: mechanics, runner, evaluator, auditor, sbatch, 26 implementation tests; CPU engineering smoke on public clips with zero auditor disagreement |
| Runtime panel / manifest | `a61dfdb2` / `fa510e45` (manifest r2 `sha256:34e4295a…`, 105 pinned sources) |
| Pre-run audit | attempt 1 BLOCK preserved (`a5b8187b`), then `PASS_TO_DIR_SPRINT0` (`62f6ee44`, 54 checks) |
| Job A | Slurm 58354, 33.0 min; seal `5d44776a` |
| Audit A | attempt 1 BLOCK preserved, plus amendment A1 (r1 auditor); `DIR_SPRINT0_AUDIT_A: PASS` (`a08ab3e9`) |
| Recost / authorization | `270dfc72`; re-serialized `6d3f1bec` (launch 58356 refused at the guard, preserved) |
| Job B | Slurm 58357, 21.5 min; seal `9c904a96` |
| PRIMARY audit (no references) | `eba309fd`: `DIR_SPRINT0_AUDIT_PRIMARY: PASS` (29 checks) |
| Evaluation | `2bac6081`; FULL audit and terminal `38c5a3de`; descriptive tables `2180cc74` |

**Model and environment.**
- Whisper-large-v3 with pinned files; BF16 and eager attention.
- Torch 2.10.0+cu128, Transformers 4.57.6.
- Weights were unchanged across both jobs, with 0 parameter gradients.
- **Phone provider:** `facebook/wav2vec2-xlsr-53-espeak-cv-ft` at 2c733782, manifest `sha256:93ae874c…`, CPU float32
  with 4 threads.
  - Job A apparatus: both children reproduced the sealed engineering posteriors **bitwise** on the compute node.
  - Audit A spot recomputation: 10/10 utterances bitwise.

## 2. D3 candidate selection and validity

**Frozen rule.**
- Inputs: float32 processed log-softmax for B0 and null audio (30 s of zeros), upcast to float64.
- Score: S = l_B + (l_B − max(l_N, log 1e-12)).
- Plausibility: l_B ≥ max l_B + log 1e-3, over the frozen 49,695-token UTF-8-legal set; lowest-ID ties.
- Direction: d_AP is the zero-probe scratch gradient of z(c_AP) − z(b_t), then the tangent.

**Structural queries (6,946):**

| Decision | Count |
|---|---|
| `candidate` (c_AP ≠ b_t) | 1,007 (14.5%) |
| `baseline_is_candidate` (exact no-edit) | 5,904 |
| `no_plausible_legal_candidate` | 35 |
| `nonfinite_branch` | 0 |

- **Window and contrast:** the plausible set has median 4 tokens (p90 61). The contrast at c_AP has median A = 4.87 nats.
  The null floor was never the deciding term.
- **Construction:** all 1,007 D3 gradient directions were valid. Every D3 scratch query matched the clean step bitwise.
- **Independent check:** the auditor recomputed every decision from the sealed logits, with 0 mismatches and 0 near-ties.

**Where the candidates point.** At the 693 structurally eligible mapped English-confusion queries, D3 proposed a
candidate 197 times (28%):

| c_AP class | Count |
|---|---|
| Han | 171 |
| Latin | 17 |
| Other | 9 |

Removing the audio-explained prior therefore mostly selects *another Mandarin token*, rarely the English word. Coverage
(constructible fraction) is 0.995 of structural queries and 1.000 of EN-confusion queries.

## 3. D4 task-state geometry and validity

- **Prompts:** transcribe `[50258,50260,50360,50364]` and translate `[50258,50260,50359,50364]`. Tokens and the task
  IDs were verified against the pinned tokenizer and `generation_config`.
- **Branches:** separate caches, identical B0 prefix and identical query index. Branch identity is 1.0.
- **Validity:** d_TF = tangent(h_TR − h_TL) was valid at **all 6,946** structural queries (coverage 1.0 and 1.0).
  ‖h_TR − h_TL‖ has median 2.52 (p10 1.73, p90 4.00).

**Task-language entanglement (disclosed, sign not changed).** Teacher-forced on the Chinese B0 prefix, the translate
branch still emits Han at 676 of 693 EN-confusion queries. Its English mass is tiny (median P_E 0.0031, against 0.0011
for transcribe). So on these prefixes "translate" barely differs from "transcribe" in output language. The TR−TL
direction is near-orthogonal to the lexical readout tangent: cos with D2 is +0.005 at EN-confusion, and the mean |cos|
of 0.028 is close to the random reference of 0.022.

## 4. D5 phone and concept coverage and prototypes

**Evidence.**
- Valid at 6,844 of 6,946 structural queries (98.5%). All 102 abstentions were `low_emitting_weight`, i.e. silence.
- Coverage: 0.985 of structural queries and 0.977 of EN-confusion queries.

**Calibration bank (FULL300, all 300 utterances, unlabeled).**
- 12,444 structural queries, of which 12,263 had valid evidence.
- All six concepts (NAS, STOP, FRIC, LAB, COR, DOR) were constructed:
  - each with 2,453+ High and Low queries;
  - with **all 20 dialogues** entering every prototype;
  - with prototype raw norms 0.08–0.12.

**Confounding diagnostics (descriptive).**
- Position |Spearman ρ| ≤ 0.03; dialogue η² ≤ 0.018.
- Cosine with the bank's baseline-token script axis (Latin minus Han states):

  | NAS | STOP | FRIC | LAB | COR | DOR |
  |---|---|---|---|---|---|
  | −0.39 | +0.08 | +0.14 | +0.41 | +0.01 | −0.59 |

  Some prototypes partly encode the output script, as anticipated in the design review.

**What the evidence shows.** The concept fractions are nearly identical across strata, for example COR 0.33–0.35 and
STOP 0.20–0.21. The language-convention signal separates the strata clearly: the tone-bearing share is 0.28 at
ZH-correct queries against 0.013 at EN-confusion queries. The provider evidently hears the English segments.

**What the direction did.** It stayed lexically inert: cos(d_PHON, D2) is −0.001 at EN-confusion, with mean |cos| 0.029.

**Shuffle.** 6,844 queries in the pool; 23 singleton utterances (23 queries), flagged and pulsed.

## 5. Population and exposure identity

**Prospective set.**
- 240 D-dev-select utterances, 12 per dialogue × 20, identity-only hash rank (selected-ID hash `sha256:5b62f378…`).
- Exclusions: all 700 registered exposures (FULL300 + SRD2-G0 400).
- 15 utterances are longer than 30 s. They were executed but are primary-unalignable.

**Baseline.**
- 240 of 240 stopped on EOS; median T = 21 (mean 31.9, max 140).
- 7,906 inventory queries:
  - 6,946 structural;
  - 720 mid-character prefixes and 240 forced t=0 queries, all recorded no-ops.

**Bank.** FULL300 (`sha256:22a160bd…`) was used unlabeled for D5 only.

**Roles untouched.** No D-dev-confirm, D-test, router-calib, P3 or transfer data.

**Exposure entry.** The 240 are now registered as exposure addendum **A2-DIR-SPRINT0-240**
(`docs/current/DATA_EXPOSURE_ADDENDA.json`; `exposed_ids()` now returns 940). `DATA_EXPOSURE.md` is untouched.

## 6. Baseline and historical matched controls (common queries, e* = 1.1260757575454359)

**Opportunities** (all frozen gates pass):

| Stratum | Mapped queries | Dialogues | Structural |
|---|---|---|---|
| EN-confusion | 701 | 18 | 693 |
| EN-correct | 297 | 16 | 286 |
| ZH-correct | 3,216 | 20 | 2,734 |

English unit mapping is 0.939 (1,885 / 2,007).

**Controls.**
- **D2 reproduces SRD2-G0.** It changes top-1 at 449 confusion queries but corrects only 2. It corrupts 275 correct
  Mandarin states, an active rate of 10.1% (SRD2 B1: 10.0%). It closes a median 2.0 nats of the confusion gap.
- **D0, D1, v_AC and random** are lexically inert (0 corrections) with small harm.
- **v_AC** was constructible at 1,766 structural queries (541 EN-confusion).
- **Old30 historical apparatus:** 30/30 bitwise in Job B.

## 7. Complete multi-arm table (all 240 rows; unique (UID, t) queries; U = C − H_ZH − H_EN)

| Arm | C (EN corr.) | H_ZH | H_EN | U | Edited ZH-correct | Active ZH rate | Severe EOS | C dialogues | ZH-harm dialogues |
|---|---|---|---|---|---|---|---|---|---|
| D0 | 0 | 10 | 2 | −12 | 2,734 | 0.004 | 0 | 0 | 10 |
| D1 | 0 | 9 | 3 | −12 | 2,734 | 0.003 | 0 | 0 | 8 |
| D2 | 2 | 275 | 1 | −274 | 2,734 | 0.101 | 25 | 2 | 19 |
| VAC | 0 | 3 | 0 | −3 | 25 | 0.120 | 0 | 0 | 3 |
| RND | 0 | 6 | 1 | −7 | 2,734 | 0.002 | 0 | 0 | 5 |
| **D3** | **2** | 80 | 6 | −84 | 96 | 0.833 | 0 | 2 | 16 |
| **D4** | **0** | 11 | 1 | −12 | 2,734 | 0.004 | 1 | 0 | 7 |
| **D5** | **0** | 8 | 0 | −8 | 2,724 | 0.003 | 1 | 0 | 7 |
| D5SH | 0 | 9 | 1 | −10 | 2,724 | 0.003 | 0 | 0 | 8 |
| D2G | 0 | 8 | 0 | −8 | 45 | 0.178 | 4 | 0 | 7 |
| D3G | 0 | 1 | 0 | −1 | 2 | 0.5 | 0 | 0 | 1 |
| D4G | 0 | 1 | 0 | −1 | 44 | 0.023 | 0 | 0 | 1 |
| D5G | 0 | 0 | 0 | 0 | 41 | 0 | 0 | 0 | 0 |
| D5SHG | 0 | 0 | 0 | 0 | 41 | 0 | 0 | 0 | 0 |
| RNDG | 0 | 0 | 0 | 0 | 43 | 0 | 0 | 0 | 0 |
| D3CD (non-steering) | 4 | 96 | 14 | −106 | 96 | 1.0 | 0 | 3 | 16 |

**The corrections.** There are only two distinct steering corrections, and both D2 and D3 made them:
- CSD0043: 因为 → "because";
- CSD0540: 或 → " or".

Both are translation-equivalent English words. D3CD added two more (CSD0043: 的 → " Good"; CSD0541: 肉 → " hamburger").
The counts are tiny and exploratory.

**Top-1 changes at the 701 EN-confusion queries:**

| Arm | Changes |
|---|---|
| D2 | 449 |
| D2G | 312 |
| D3CD | 197 |
| D3 | 193 |
| D3G | 161 |
| D1 | 39 |
| VAC | 34 |
| D5SH | 29 |
| D4 | 27 |
| D0 | 26 |
| D5 | 21 |
| RND | 17 |

Only D2 and D3 move the decision often, and almost always to a non-acceptable token.

**Every family-variant predicate failed** (energy validity passed throughout):

| Variant | Failed predicates |
|---|---|
| D3, D4, D5 (ungated) | power, advantage, safety |
| D4G, D5G | power, advantage, safety (estimable: 44 and 41 edited ZH-correct states) |
| D3G | power, advantage, safety, estimability (2 edited ZH-correct states) |

## 8. R2-gated versus ungated

**Exposure.** The frozen R2 `g_old` was positive at 1,554 structural queries (22%, all 20 dialogues). Gating cuts
edited ZH-correct exposure from 2,734 to 41–45 states, so Mandarin harm falls to 0–8 events.

**No lexical gain in any gated arm.**
- Every gated arm makes 0 corrections, while the ungated arms make 2, 0 and 0 for D3, D4 and D5.
- R2-gated D2 (D2G) closes a median 1.5 nats of the confusion gap and corrects nothing. SRD2-G0 B2 found 1 correction
  on a 400-utterance population.

**Planned-energy loss.** Gated arms lose at most 0.051% of planned squared energy (tiny gates are unattainable at native
BF16). These are explicit no-edits, reported, not substituted.

## 9. Realized edit energy

- 58,578 executed cells. Every ungated arm matched **1.000** of its valid-direction requests.
- Maxima: relative squared-energy error 1.39% (frozen 2%), chord error 0.69% (2%), consumed/proposed 0.498% (0.5%).
- Every executed cell:
  - FFN input bitwise equal to the native preview;
  - one solve;
  - solver = hook;
  - restore bitwise.
- Clean queries are bitwise equal to Job A at all 7,906 queries. D2 and D3 scratch identities hold everywhere.

## 10. EOS and Mandarin safety

**Severe EOS proxy** (new EOS top-1 with ≥ 10 baseline tokens remaining):

| Arm | Events |
|---|---|
| D2 | 25 |
| D2G | 4 |
| D4 | 1 |
| D5 | 1 |
| All other arms | 0 |

**Mandarin harm.**
- D2 is the only arm with large harm: 275 corruptions, active rate 10.1%, a damage dialogue present.
- **D3 is unsafe where it acts.** It edits only 96 correct-Mandarin states (those where the contrast prefers another
  token), and 80 of them flip (83%), because its target is by construction not the correct baseline token. Its utility
  is −84 against D3CD's −106.
- D4 and D5 corrupt 11 and 8 ZH-correct states at 0.3–0.4% unconditional rates, at the level of D0, D1, RND and the
  shuffle. They are mostly harmless because they barely move the decision.

## 11. Phonetic shuffle and contrastive-decoding comparisons

**D5 against D5SH (shuffled evidence).**

| Comparison | D5 | D5SH |
|---|---|---|
| Corrections | 0 | 0 |
| ZH corruptions | 8 | 9 |
| Gap closure beyond random | +0.012 nat | similar |
| cos with D2 | ≈ 0 | ≈ 0 |

D5 is indistinguishable from its own control: local acoustic phonology adds nothing to the decoder's lexical choice at
e*. Gated, D5G and D5SHG both have 0 corrections and 0 harm.

**D3 steering against direct contrastive selection.** D3CD selects c_AP outright:
- 4 corrections, 96 ZH and 14 EN corruptions, utility −106;
- steering D3 attains 2 corrections, 80 and 6 corruptions, utility −84.

Steering moves the top-1 toward c_AP less reliably than selecting it, and harms correspondingly less. But the candidate
itself is the problem: it is the acceptable English token in only 4 of 701 confusions.

## 12. Dialogue-level uncertainty

**Bootstrap.** Paired dialogue-cluster bootstrap: 10,000 draws, seed 240924, the same draws for every arm.

**Corrections:**
- 95% intervals: [0, 5] for D2 and D3, [0, 9] for D3CD, [0, 0] for every other arm.
- Primary family (m = 30, Bonferroni quantiles 0.05/60): every correction contrast against matched D2, RND or the
  shuffle includes 0 or is [0, 0]. For example C_D3 − C_RND is [0, 7], C_D4 − C_D2 is [−7, 0] and C_D5 − C_D5SH is
  [0, 0].

**Utility:**
- U_v − U_D2 is positive for D3, D4 and D5 only because D2 is so harmful (e.g. U_D4 − U_D2 [152, 386]).
- Against random, U_D3 − U_RND is [−118, −39]; D4 and D5 against random include 0.

**Mechanistic.** Δgap beyond matched random, Bonferroni m = 6, in nats:

| Variant | Point | Interval |
|---|---|---|
| D3 | −0.20 | [−0.39, 0.11] |
| D3G | −0.14 | [−0.24, −0.06] |
| D4 | +0.010 | [−0.022, 0.037] |
| D4G | +0.014 | [−0.005, 0.032] |
| D5 | +0.012 | [−0.013, 0.036] |
| D5G | +0.005 | [−0.010, 0.024] |

None reaches the frozen 0.5-nat point with a lower bound above 0.

**Concentration.** Steering corrections occur in only 2 dialogues (CSD0043, CSD0540).

## 13. Audit results

| Stage | Result |
|---|---|
| Pre-run | attempt 1 `BLOCK_BEFORE_DIR_SPRINT0`, preserved |
| Pre-run (rerun) | `PASS_TO_DIR_SPRINT0` |
| Audit A | attempt 1 `BLOCK`, preserved |
| Audit A (r1, amendment A1) | `DIR_SPRINT0_AUDIT_A: PASS` |
| PRIMARY (no references) | `DIR_SPRINT0_AUDIT_PRIMARY: PASS` |
| FULL | `DIR_SPRINT0_AUDIT_FULL: PASS` |

**Pre-run attempt 1.** The only failure was the constant screen matching `"transcript"` inside the Whisper token string
`<|startoftranscript|>`. Only the auditor rule changed (exact special-token strings are exempt), and the manifest was
rebuilt, since no job had used it.

**Audit A attempt 1.** The only failure was opened-path classification:
- a hash-only pinned historical source with a reference-like name (`inference_cf_p2r_population.py`; SRD2-G0 A1
  precedent);
- the provider directory name containing "espeak".

Amendment A1 resolves explicit allowlist categories before the name screen and lists hash-only pinned files separately.
Every recomputation passed in both attempts. The canonical auditor is kept at its manifest bytes; `run1/audit_r1/` is
the auditor of record (`AMENDMENT_A1.json`).

**Launch 58356.** Refused after 4 s by the runner's authorization field-order guard, before model load or any data
access. The cause was the sorted-key serialization noted in SRD2-G0. It is preserved; the content was re-serialized in
the frozen order and relaunched as 58357.

**What the FULL audit reproduced independently:**
- mapping (SRD2 auditor `own_map`);
- every arm count and per-dialogue tally;
- opportunities, coverage and statuses;
- every Bonferroni interval (agreeing within 1e-9);
- the mechanistic points;
- the label.

## 14. Terminal label and per-family conclusions

**First match: `DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE`.**

| Rule | Result |
|---|---|
| INVALID | No critical failure; no family blocked |
| COMPUTE_BLOCKED | Recost passed (5,017 / 3,648 s) |
| OPPORTUNITY_INSUFFICIENT | All gates pass; all families VALID_AND_TESTED |
| FEASIBILITY_SIGNAL / CAUSAL_POWER_WITH_DAMAGE | No family reached power + advantage |
| MECHANISTIC_EFFECT_ONLY | No mechanistic effect |

**D3: VALID_AND_TESTED / LEXICAL_POWER_INSUFFICIENT.**
- The candidate rule is well defined and reference-free, and it aligns with the lexical readout where it acts
  (cos(d_AP, D2) +0.35 at EN-confusion).
- But removing the language-model prior points to another Mandarin token in 87% of confusion candidates.
- It makes the same 2 corrections as D2, and it is unsafe where it acts: 83% of the correct-Mandarin states it edits
  flip.

**D4: VALID_AND_TESTED / LEXICAL_POWER_INSUFFICIENT.**
- On teacher-forced Chinese prefixes, transcribe and translate states differ little in output language. TR−TL is
  lexically orthogonal (|cos| ≈ random).
- It makes 0 corrections. Its harm is minor, at random level.

**D5: VALID_AND_TESTED / LEXICAL_POWER_INSUFFICIENT.**
- The provider and evidence work: 98.5% coverage, and the evidence plainly separates English from Mandarin audio.
- But the concept-prototype direction is lexically orthogonal: it makes 0 corrections, is indistinguishable from its
  shuffled control, and is partly script-confounded.

The negative results are scientific outcomes, not provider or construction failures.

## 15. Is any direction worth subsequent full decoding?

**No.** No new family meets the frozen power bar (≥ 8 corrections in ≥ 4 dialogues) or beats D2 or random. Two
families (D4, D5) are lexically inert, and the third (D3) shares D2's two corrections at much higher active Mandarin
harm. Full autoregressive decoding would only compound single-query damage. This result does not authorize full ASR
decoding, A2-TTA integration, test-role use or transfer.

## 16. Close the steering branch and return to A2?

**Yes: close DIR-SPRINT0 and the steering-direction-discovery branch.** Across MECH-LANG0, P2-DIR, ST-LOC0, SRC-CF0-P,
SRD2-G0 and now DIR-SPRINT0:
- prompt, subspace, acoustic, task, phonetic and random directions at the L16 DG-02 site and e* are lexically
  orthogonal;
- the readout-gradient family (D2, D3) moves decisions but corrects only near-boundary confusions, and damages
  Mandarin at about 10% (D2) to 83% (D3, where active).

There is no automatic second direction-discovery programme. The most justified next step remains the separately frozen
main-line exact-A2 confirmation (P2-A2-MECH0 `A2_CONFIRMATION_READY = YES`). That confirmation requires its own explicit
human authorization; nothing here starts it.

## Limitations

- **Same dialogues.** New utterances from the same 20 already-used dialogues, and the bank shares those dialogues. There
  is no independent-dialogue or speaker generalization.
- **Single-query, first-token outcomes.** The EOS proxy is not a measured truncation. Mandarin endpoints are next-token
  preservation.
- **Gated variants** have only 41–45 edited correct-Mandarin states (D3G: 2), so their safety evidence is thin.
- **D5 depends on an external phone model** whose conventions confound voicing and aspiration (excluded from the primary
  concepts). Its prototypes partly encode output script.
- **One frozen dose and site.** Nothing here tests other layers, doses or full decoding.
- **15 long utterances** are primary-unalignable.

## Artifacts

**Run directory:** `results/inference_cf/dir_sprint0/run1/`, containing:

| Group | Files |
|---|---|
| Manifests and audits | manifest r2, `audit_PRE`, `audit_A` (+ r1), `audit_PRIMARY`, `audit_FULL` |
| Seals | `job_A_seal`, `pulse_seal` |
| Authorization | `resource_forecast`, `authorization_B` |
| Results | `evaluation`, `terminal`, `descriptive` |
| Preserved failed attempts | `audit_PRE_attempt1_BLOCK`, `manifest_attempt1_superseded`, `audit_A_attempt1_BLOCK`, `job_B_launch_58356_guard_refusal`, `authorization_B_runner_sorted` |

**Raw arrays** (9.8 GB, 1,801 files) are archived content-addressed under
`/mnt/data/tungnx/cs-asr-steer/archives/dir_sprint0/run1`. `evaluation_units.json` (reference surfaces) is local only;
its hash is in `evaluation.json`.

**Design:** `docs/inference_cf/DIR_SPRINT0_{SPEC,DESIGN,FIREWALL,COMPUTE_FEASIBILITY}.md`, `DIR_SPRINT0_POPULATION.json`,
`DIR_SPRINT0_FREEZE.json`, `configs/inference_cf/dir_sprint0.json`.

**Exposure:** `docs/current/DATA_EXPOSURE_ADDENDA.{json,md}` (A2).

**Unchanged:** core v6, METHOD_CONTRACT and every historical verdict and result.
