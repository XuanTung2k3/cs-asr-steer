# P2-PATH0 — First-Divergence Rescue / Induction Causal Test

Pre-outcome freeze after audited `P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED`. Starting local/remote HEAD `b5031e998b5ce0b135eb6f524e5e8df9bf8e9a80`, branch `feature/inference-cf-steering`, clean tree, empty user Slurm queue. This is a bounded exposed-development causal decoding diagnostic; core v6 is unchanged. A2/A3/A4 stay closed and byte-preserved. No PATH0 scientific outcome, adaptation or clamp inference occurred in the design session.

Normative companions: `configs/inference_cf/p2_path0.json`, `P2_PATH0_SITES.json`, `P2_PATH0_CODEX_DESIGN.md`. All sites derive solely from sealed token arrays/termination metadata, never reference quality. The config anchors inspected historical sources and A4 scientific settings. Implementation must add a separate path, not modify historical A2/A3/A4.

## Question and scope

Does the first branch decision, or its next three decisions, causally control A4's subsequent decoding path? Teacher-forced English transfer and non-English anchoring succeeded, but Mandarin safety did not. PATH0 tests whether the downstream suffix can be rescued toward B0 or induced toward AUTO while holding the exact adapted A4 weights fixed. It does not select a teacher/objective/gate or claim a deployable rescue policy.

## Population and sites

Only the12 D rows in the exact A3/A4 panel: theta0 AUTO text differs from FORCED as historically frozen. Reproduced content-token/termination comparison gives **C=5 RESCUE**, **U=7 INDUCE**. Do not include12 A controls or reselect using errors. Parent panel byte hash `6422741b91e64ff007458b7a04d2f559e8ed8ffb592a087a538adf63632b2206`.

- C: first differing decision between sealed A4 and B0. Before it both token prefixes must match exactly. Donor B0.
- U: A4 tokens/termination equal B0. First differing decision between B0 and AUTO. Before it their prefixes match. Donor AUTO.

Sites are zero-based generated-content decision indices; fixed Whisper prompt never enters this index. The JSON freezes every row, dialogue, role, index, common-prefix IDs/hash, B0/alternative token, donor1/3 continuation, effective lengths, source tokens, termination, suffix eligibility, row/checkpoint hashes. No reference fields are present.

## Exact token, EOS, cap and hashing semantics

Historical `forced_decode` saves `tokens` without prompt or terminal EOS, plus `terminated`=`eos` or `cap`. A3/A4 plan's y_B/y_A are already cleaned generated-content arrays, not text retokenizations. Preserve UTF8 fragment token IDs and spaces exactly.

For divergence and suffix edit distance define decision stream `s(y)=tokens+[50257]` iff terminated=`eos`; otherwise just tokens. The added terminal EOS represents a real decoder decision, not a generated content word. Never append an EOS to a capped sequence or count the fixed prompt. Runtime serialization retains the historical content-only array and separate termination flag. Logical EOS permits strict-prefix divergence to be diagnosed correctly: shorter EOS-terminated stream differs from longer stream at its EOS decision. Cap-truncated strict prefix with no available decision at k is an undefined clamp site → INVALID before run, not fabricated EOS or alternate site. Actual12 rows are all EOS-terminated with real tokens available at each site.

Unexpected interior prompt/timestamp/control token (IDs≥50257 in these historical content arrays) is INVALID; do not silently strip it. EOS is permitted only as the appended logical terminal decision. Preaudit validates this against actual tokenizer control IDs and frozen no-timestamps semantics. All12 observed source arrays contain no such controls.

For requested L=1 or3, force donor decision stream `[k:min(k+L,200)]`, truncated at its terminal EOS/end. Effective L is the count of actually forced decisions, including EOS if forced. No continuation is invented after EOS/cap. All frozen clamps here have effective lengths1/3; `U1003_S0_61` INDUCE-3 includes EOS as its third decision. Its released suffix is empty and its baseline denominator0: DIRECT_ONLY.

A forced or naturally selected EOS stops immediately. Cap stays200 total next-token decisions from prompt, including forced decisions and any EOS decision; it does not restart after release. Any requested clamp beyond budget is shortened in the sealed artifact; no runtime adjustment except exactly stopping on its sealed EOS. Hash token arrays/common prefixes using ASCII JSON `separators=(',',':')`, no trailing newline, SHA256; file hashes use exact UTF8 bytes including final newline.

## Exact A4 reconstruction and all-row validity barrier

Reuse unchanged `adapt_a4`, A4 plan, teachers/masks/classes,194 LN tensor list, AdamW1e-3/wd0/betas0.9/0.999/eps1e-8,2 steps, fp32 masters/bf16 forward, original no-op and reset. Original native AUTO detection must equal sealed A3/A4 lang ID; no teacher regeneration. Encode each of12 original full utterances once, detached as in A4, no future token/reference access. Reuse the encoded tensor for reconstruction and all branch decodes.

At theta0, ordinary forced decode must reproduce sealed B0 tokens and termination12/12. Record theta0 site logits through that exact cached token-by-token path. Adapt exactly once per row, save effective state, materialize, run ordinary FREE-A4; require sealed A4 tokens AND termination12/12, and reconstructed effective bf16 LN tensors bitwise equal the sealed historical fp32 final-master checkpoint cast to bf16. Verify non-LN immutability, finite parameters and theta0 reset. Loading the old checkpoint instead of performing the two updates is forbidden; it is an integrity comparator only.

**Complete all12 reconstructions/FREE checks before any scientific clamp.** Keep detached encoders and tiny reconstructed effective states for phase2; do not adapt a second time. One live independent first-row A4 formula check may reuse the historical audit routine (loss1e-5, relative gradient2%, floor1e-8), with no extra update. Thus24 optimizer steps and at most25 backwards. Any failure → P2_PATH0_INVALID, stop before clamps. No A4 correction or replacement baseline.

## Clamp decoding

Use the same A4 forced-ZH prompt `[50258,50260,50360,50364]`, bf16/eager/eval/batch1, passive capture L16, attention=True, original processed_argmax/suppress/begin-suppress and greedy cap200. No steering hook, D2, new prompt, beam or sampling.

Each FREE/SELF/1/3 call starts a fresh `cached.Branch` and KV cache; feed full prompt once, then one chosen token each step. Do not batch-prefill the common prefix, copy a mutated cache or use a teacher-forced full-prefix path. Greedy before k must equal the sealed common prefix; then override only k..k+effectiveL−1 with the sealed donor IDs. Each override must be allowed by suppression at its absolute content index; disallowed token → INVALID, never change suppression. Feed the forced token into that branch before computing the following decision. After release use ordinary A4 greedy with no additional constraints.

- RESCUE rows: FREE-A4, RESCUE-1(B0 donor), RESCUE-3(B0 donor).
- INDUCE rows: FREE-A4, INDUCE-1(AUTO donor), INDUCE-3(AUTO donor).
- SELF: force exactly FREE-A4's own decision at k (B0 for U), release. Must reproduce full FREE content IDs and termination on12/12; engineering-only, no causal credit.

## Reference-free suffix endpoints and pass rules

Use unit-cost token-ID Levenshtein, including the terminal EOS decision stream as above. For each arm let b=k+effectiveL. **Exclude the entire forced span by slicing every compared stream at b**, even if their lengths differ. Do not align the intervention to a different release index.

RESCUE:
`d0=ED(FREE[b:],B0[b:])`, `d1=ED(RESCUE[b:],B0[b:])`, `R=1-d1/d0` when d0>0.

INDUCE:
`d0=ED(B0[b:],AUTO[b:])`, `dA=ED(INDUCE[b:],AUTO[b:])`, `dB=ED(INDUCE[b:],B0[b:])`, `R=1-dA/d0` when d0>0.

Denominator0 → DIRECT_ONLY for that L; exclude from pass counts AND aggregate numerator/denominator, report full output descriptively. Ratios may be negative; do not clip. Eligibility is sealed from historical source streams and never redefined from clamps. Current denominators: RESCUE1/3 each4; INDUCE1=6, INDUCE3=5. Each criterion requires at least3 eligible rows; otherwise it fails without scientific fallback.

| Criterion | Row success | Required successes | Pooled reduction | Dialogue support |
|---|---|---|---|---|
| RESCUE_1_PASS | R≥0.50 | ≥3/4 (strict majority) | ≥0.50 | successes span≥2 dialogues |
| RESCUE_3_PASS | R≥0.50 | ≥3/4 (strict majority) | ≥0.50 | successes span≥2 dialogues |
| INDUCE_1_PASS | R≥0.50 AND dB>0 | ≥3/6 | ≥0.30 | successes span≥2 dialogues |
| INDUCE_3_PASS | R≥0.50 AND dB>0 | ≥3/5 | ≥0.30 | successes span≥2 dialogues |

Pooled reduction is `1−sum(intervention target distance)/sum(baseline target distance)` over **all** eligible rows, not only successes. Generic count rules: rescue floor(n/2)+1; induction max(3,ceil(n/2)). No significance requirement or fitted threshold. Fifty-percent row reduction is substantial path movement; pooled rescue50% guards against a token-only repair, and induction30% plus3 successes/2 dialogues guards against one exceptional utterance. Thresholds are chosen from the frozen small cohort/eligibility and causal meaning, not clamp outputs. Float guard1e-12.

## Terminal labels and claim boundaries

Exact precedence:

1. Any population/hash/site/reset/reconstruction/SELF/suppression/cache/finite/complete-output/audit failure → **P2_PATH0_INVALID**.
2. RESCUE_1_PASS AND INDUCE_1_PASS → **P2_PATH0_SINGLE_TOKEN_CAUSAL**.
3. Else RESCUE_3_PASS AND INDUCE_3_PASS → **P2_PATH0_SHORT_PREFIX_CAUSAL**.
4. Else any of the four PASS → **P2_PATH0_ASYMMETRIC**; report which direction/length passes.
5. Else → **P2_PATH0_NO_LOCAL_BRANCH_CAUSALITY**.

Single-token evidence means these interventions are sufficient to influence later suffixes under fixed adapted A4 state on this cohort, not a universal necessity proof. Short-prefix means matched one-token criteria failed but the three-decision intervention establishes bidirectional suffix control; report effective lengths/EOS explicitly. Asymmetry does not prove a unique basin. No-local means neither one/three clamp establishes the preregistered downstream effect; distributed/earlier failure remains a hypothesis, not a demonstrated cause. References cannot change these labels.

## Branch-state diagnostics

At the exact common prefix, capture theta0 forced and reconstructed A4 processed float32 logits from the matched cache path. Candidate B=B0; X=FREE-A4 for rescue, X=AUTO for induction. Record `z(X)−z(B)`, probability X/B, canonical P_E/P_M masses, entropy and adapted-minus-theta0 margin shift. Probabilities/entropy use temperature1 on the unchanged finite allowed vocabulary; logits use the same suppression mask (both candidates must remain finite). Same canonical A4 `whisper_han_ascii_v1` partition. This is descriptive only and cannot change sites, clamps, eligibility or thresholds. Save compact scalars, not vocabulary tensors.

## Secondary ASR evaluation after primary seal

Commit/push all clamp outputs, source/parameter/cache checks, suffix distances, PASS booleans and reference-free primary label inputs/label **before loading references**. Independent primary audit may be performed before references; final audit covers both phases. Later load only the12 already-exposed references.

Report per row and by role/arm: mixed errors, matrix-ZH errors/ZH-CER contribution, embedded-EN/POI errors, substitutions/deletions/insertions and matrix retention where defined. Rescue compares FREE/R1/R3/B0; induction compares FREE/I1/I3/AUTO. Whole-transcript metrics include the forced span and therefore are secondary; they cannot establish suffix causality themselves.

Frozen descriptive ASR_HARM_ALIGNMENT:

- Rescue-aligned iff **one same L** passes RESCUE_L and, over that L's sealed eligible C rows, aggregate ZH error count falls≥2 versus FREE, at least2 rows individually reduce ZH errors, and aggregate POI error count does not increase.
- Induction-aligned iff **one same L** passes INDUCE_L and, over that L's sealed eligible U rows, aggregate ZH errors rises≥2 versus FREE/B0 and at least2 rows individually increase ZH errors.
- Both → BIDIRECTIONAL; rescue only → RESCUE_ONLY; induction only → INDUCTION_ONLY; neither → NONE. Do not pool1/3 successes to pass a flag. Empty reference language populations report not assessable/flag false, no row replacement. Use canonical count extraction, no new evaluator.

No formal confidence claim from5/7 rows. Report row counts/dialogues and pooled distances. Optional uncertainty is omitted from primary decisions; no post-outcome statistical criterion.

## Compute, firewall and audits

One sbatch H100/MIG job maximum,12 rows, target<10min, hard30min, at most one pending/running PATH0 job.12 theta0 integrity decodes +12 FREE +12 SELF +24 scientific clamp decodes =60 decode passes maximum;24 updates/≤25 backwards, no new objective or full24/100/300 study. Budget exhaustion/missing row → partial INVALID, STOP; no automatic second job.

Allowed exact12 D from exposed D-dev-select and sealed A3/A4/B0/AUTO artifacts. No fresh validation, D-dev-confirm/D-test/router-calib new role, transfer corpora, SEAME/CS-FLEURS/ViMedCSS/ASCEND, full100/300/P3. Source transcripts are diagnostic interventions, not inference-time deployment features. Reference labels never construct row/site/donor/length/role.

Pre-run independent **PASS_TO_P2_PATH0** required and pushed with resolved config/environment/git/model/input manifest using existing provenance conventions. Post-run **P2_PATH0_AUDIT: PASS** required before conclusions. Auditor independently recomputes C/U, streams, sites/hashes, donor spans/effectiveL/eligibility, reconstruction/SELF/reset, suffix ED, all four PASS, label, firewall and secondary alignment; must not import primary decision/site-selection code. Canonical evaluator primitives may be shared.

Freeze→focused tests→diff review→commit→push before outcome. Later implementation/manifest/preaudit and primary output seal follow the same GitHub discipline; no force/main merge/PR. Stop after the audited PATH0 report regardless of label. No automatic follow-on experiment is authorized.
