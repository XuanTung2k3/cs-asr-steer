# P2-PATH1 — Branch Adjudication and State-vs-Prefix Causal Decomposition

Pre-outcome freeze following audited PATH0 ASYMMETRIC (INDUCE1/3 pass; rescue fails; ASR induction-only). Starting local/remote HEAD `42ae992eaebabf1f57cf1f06b961e4608bef091e`; branch `feature/inference-cf-steering`, clean tree and empty user queue. No PATH1 inference/adaptation/outcome or new reference evaluation in this design session. This is separate exposed-development diagnosis; core v6 and historical A2/A3/A4/PATH0 remain unchanged.

Normative artifacts: `configs/inference_cf/p2_path1.json`, `P2_PATH1_SITES.json`, this spec and `P2_PATH1_CODEX_DESIGN.md`. No new objective, gate, LR/step/subset or horizon search.

## Primary question and population

Is the one-token AUTO branch sustained by history alone under theta0, or does persistent A4 state add the necessary continuation bias? Exactly the seven PATH0 U/INDUCE rows, all with A4=B0 before intervention; first B0/AUTO sites inherited without change. Six suffix-eligible U rows, one DIRECT_ONLY (`U1079_S0_23`). All seven have k=0, so this specific test concerns an initial branch decision, not a long inherited prefix. Keep all7 outputs, never replace/drop rows.

The new site file fingerprints the entire parent PATH0 site file and each original row (sorted-key compact JSON SHA256), copies exact common prefixes/candidates/L1 eligibility and points to historical source/termination hashes. It adds only frozen H=3 donor continuations and PATH0 A4A reproduction targets. Five C/RESCUE rows are **short-score-only** supporting population; no new rescue decodes or scientific C factorial. Twelve rows maximum, from the same exposed D-dev-select panel.

## Streams, prompts, suppression and cache

Inherit PATH0 exactly: stored generated content IDs exclude fixed prompt and EOS; metric streams append one logical50257 iff terminated=eos, never at cap. Divergence/release indices count those decisions; retain content IDs including UTF8 fragments, no text retokenization. Unexpected controls, missing decision or incompatible cap/EOS metadata → INVALID. Same forced-ZH prompt `[50258,50260,50360,50364]`, bf16/eager/eval/batch1, token-by-token `cached.Branch`, passive L16 capture/attention=True, generation suppression with begin-suppression only at absolute content index0, greedy cap200 decisions including forced token. No steering/D2.

**No cross-model decoder cache.** Each model×branch path starts a fresh Branch after materializing/confirming its selected state. Replay common prefix under that same state, force the selected token, advance/cache that token under that state and continue greedily under that state. Encoder outputs are frozen and may be shared, decoder KV may not. Destroy the cache before restoring or switching LN state. No copied checkpoint cache, cross-state prefill, batched common-prefix shortcut, or switching weights after a forced token. State hash stays constant for the entire path. Pre-site greedy decisions must equal the sealed prefix.

## Reconstruction and four U conditions

Reconstruct A4 exactly once per needed utterance (all12 because both models score C): unchanged adapt_a4,194 LN affine tensors/248320 scalars, sealed y_A/mask/classes/native condition, two fresh AdamW updates at1e-3, wd0, original precision/reset/no-op. Reuse original full audio/encoder once per row; detached clone for adaptation. Before factorial/scoring, all12 theta0 B0 replays and FREE-A4 replays must match sealed tokens/termination and effective bf16 LN tensors must equal historical sealed final masters cast bf16. All-row barrier; any mismatch INVALID, no correction to A4. Original independent first-row loss/gradient check allowed without extra update (1e-5/2%, floor1e-8). Retain tiny effective states/encoders, reset theta0 exactly, no episode transfer.

Run **all four new** U decodes, not replacement by cached predictions:

| Condition | State for entire prefix/cache/continuation | Forced token |
|---|---|---|
| T0B | theta0 forced-ZH | c_B |
| T0A (prefix-only) | theta0 forced-ZH | c_A |
| A4B | reconstructed A4 forced-ZH | c_B |
| A4A (prefix+adapted) | reconstructed A4 forced-ZH | c_A |

Force one decision at k only, then release immediately. T0B/A4B must reproduce sealed B0 on7/7; A4A must reproduce PATH0 L1 content IDs and termination on7/7. These are mandatory integrity checks; A4A must also independently pass original INDUCE_1 criterion. Failure → P2_PATH1_INVALID, STOP. Historical PATH0 branch diagnostics can be reused after hash/state/site/suppression audit; no need to regenerate them solely for convenience.

## Exact primary endpoints and inherited criterion

Release r=k+1 on each row. Use PATH0 unit-cost token ED on decision streams sliced at r:

`d_BA=ED(B0[r:],AUTO[r:])`;
`d_mA=ED(mA[r:],AUTO[r:])`;
`d_mB=ED(mA[r:],B0[r:])`.

Eligible iff d_BA>0. `I_m=1-d_mA/d_BA`; DIRECT_ONLY otherwise, None for attraction/increment, excluded from all primary aggregates and mechanism counts. Do not clip negative values. Forced token receives no credit.

Apply exact PATH0 INDUCE_1 independently to T0A and A4A: success I≥0.50 and d_mB>0; ≥3 of6 eligible successes (generic max(3,ceil(n/2))), ≥2 success dialogues, pooled attraction≥0.30, and at least3 eligible rows. Pooled I is `1−sum(d_mA)/sum(d_BA)` over all eligible rows, including failures. Frozen parent sum(d_BA)=48. Primary checks must reproduce parent denominators and A4A's exact outputs, not merely similar pooled values.

`Delta_state_row=I_A4−I_theta0`; `Delta_state_pooled=I_A4_pooled−I_theta0_pooled`. These are paired same-row state effects with fixed forced token/history intervention.

## Frozen mechanism criteria

For eligible row i define materiality `tau_i=max(0.25,1/d_BA_i)`: one suffix edit at minimum and a quarter-baseline reduction for longer rows. Approximate equality: abs(Delta_i)<tau_i. Material A4 amplification: Delta_i≥tau_i. Report theta0-stronger rows separately when Delta_i≤−tau_i. Substantial theta0 return toward B0: I_theta0≤0.20 AND d_theta0B≤max(1,floor(0.25*d_BA)). These counts are descriptive; no reference enters them.

The following rules are inclusive with float guard1e-12; approximate equality uses strict inequality. No formal significance demand for6 eligible rows. The20/30/25% thresholds are fixed materiality choices from discrete suffix resolution and cohort size, not PATH1 outcomes.

Exact precedence:

1. Any integrity/audit failure, including A4A reproduction/pass failure → **P2_PATH1_INVALID**.
2. **P2_PATH1_PREFIX_DOMINANT** iff T0A passes inherited INDUCE_1 AND Delta_pooled≤0.20 AND at most2 eligible rows have material A4 amplification. Persistent A4 is unnecessary to establish the basin on this cohort; it may still amplify some rows.
3. **P2_PATH1_ADAPTED_STATE_DOMINANT** iff T0A fails inherited INDUCE_1 AND I_theta0_pooled≤0.20 AND T0A row successes≤1 AND Delta_pooled≥0.30 AND ≥3 materially amplified rows spanning≥2 dialogues. This supports a necessary contribution from persistent A4 state for strong induction under this intervention/cohort, not universal necessity.
4. **P2_PATH1_MIXED** iff any of:
   - T0A passes inherited criterion but fails the prefix-dominant bounds (material amplification/heterogeneity);
   - T0A pooled I≥0.30 and ≥2 rows have I_theta0≥0.25 AND d_theta0B>0, spanning≥2 dialogues, but neither dominant label applies;
   - ≥2 T0A inherited row successes AND ≥2 **other** state-required rows (A4 row success, not T0 row success, Delta≥tau), with each set spanning≥2 dialogues, but neither dominant label applies.
5. Otherwise **P2_PATH1_NO_CLEAR_MECHANISM**.

Mixed captures meaningful prefix attraction or row-specific dependence without forcing dominance from one large-distance row. No-clear is valid insufficient/contradictory evidence. Report all booleans/counts and signed increments, not just label. No mechanism classification from the DIRECT_ONLY row. Primary mechanism labels describe AUTO-like suffix attraction, not ASR correctness; the separately frozen ASR flag tests whether that attraction carries actual Mandarin harm.

## Secondary fixed branch adjudication

H=3 **content tokens**, no horizon sweep. At each of12 parent sites, B donor is sealed B0 content `[k:k+3]`; ALT donor is sealed AUTO for U, sealed FREE-A4 for C. H_eff is each branch's available content count, independently capped at3. Unlike PATH0's three-decision clamp, EOS is not a scored content token; near EOS the horizon shortens. All actual branches have≥1 content token; missing/invalid branch → INVALID, no replacement/fabricated EOS. No full continuation generation for scoring.

For each model and branch teacher-force the donor through the same token-by-token cache path, with prompt/common prefix under that model, then score each donor token at absolute content index k+j before feeding it. Use float32 log_softmax of canonical generation-processed logits, temperature1. Suppressed donor or nonfinite score → INVALID. No whole-prefix `use_cache=False` alternative (bf16 near-tie semantics may differ). Fresh cache for **every** model×branch, no greedy completion needed, no gradients or adaptation updates while scoring.

`S_m(b)=sum_j log p_m(b_j|common_prefix,b_<j)/H_eff(b)`;
`S_cons(b)=0.5*S_theta0(b)+0.5*S_A4(b)`.

Choose larger consensus score; abs difference≤1e-12 chooses B. `S_min=min(S_theta0,S_A4)` descriptive only, cannot replace consensus. Save per-token log probabilities, branch lengths, both scores/margin/choice and state/cache ownership traces. Mean-length normalization permits comparison of near-EOS branches but does not evaluate termination likelihood; disclose this limitation.

**CONSENSUS_REJECTS_AUTO** iff among the6 suffix-eligible U rows, ≥5 choose B, ≥4 are strict B wins (consensus difference>1e-12), and strict B wins span≥3 dialogues. DIRECT_ONLY U is reported outside this criterion. This is a reference-free rejection screen for a branch known to cause cohort-level harm under A4; it does not assume every U row is individually harmed and is not an ASR improvement claim or deployed beam-search algorithm. C choices are descriptive only, never tune weights/horizon/tie rule or replace score. An unscored continuation cannot be called a lower-error hypothetical transcript.

## Reference firewall and secondary evaluation

Before opening references, immutable seal includes all28 factorial outputs, A4/B0 reconstruction checks, state/cache ownership, suffix ED/attractions/increments, complete primary criteria/label inputs and reference-free label, all48 score paths/choices, manifest/config/sites/input hashes. Commit/push first. Independent primary verification may precede reference access; failed primary validity blocks references.

After seal evaluate seven U systems B0/AUTO/T0A/A4A: ZH/POI/mixed errors, D/I/S, lengths/EOS/caps and group/per-row counts. Report AUTO excess harm and POI gains separately; forced token can affect these whole-transcript secondary metrics and gets no primary causal credit. Descriptive C comparison: report scores/choice alongside sealed B0/FREE-A4 whole-transcript errors, explicitly not a simulated adjudicated continuation.

Frozen **ASR_STATE_ALIGNMENT**, computed on the same6 primary eligible U rows only: define Harm_m iff ZH-error excess versus B0≥2 in aggregate AND at least2 rows individually increase ZH errors. Let h_m be signed aggregate excess.

- PREFIX iff Harm_theta0 AND Harm_A4 AND h_A4−h_theta0≤2.
- Else ADAPTED_STATE iff Harm_A4 AND NOT Harm_theta0 AND h_A4−h_theta0≥3.
- Else MIXED iff either Harm holds.
- Else NONE.

POI gains/losses are reported separately, not used to mutate this harm flag or primary mechanism label. Empty ZH population yields not-assessable/NONE; no refilling. Canonical evaluators/counts unchanged. No significance claim on this small selected cohort.

## Compute, audits and stop

One sbatch H100 MIG job maximum, target<10min, hard30min, at most one pending/running. Twelve once-only A4 reconstructions:24 updates/≤25 backwards;12 theta0 sanity+12 FREE-A4+28 factorial decodes=52 full decodes;48 short score paths≤144 target log-prob terms (common-prefix replay queries additional). No scientific C factorial, new L3 clamp or new adaptation objective. No extra jobs for partial/budget failure; INVALID STOP.

Require independent **PASS_TO_P2_PATH1** committed/pushed with resolved manifest before job, and **P2_PATH1_AUDIT: PASS** before conclusions. Auditor must not import primary decision code: independently verify parent membership/sites/tokens/termination, all four state×branch paths, no stale cross-state KV, T0B/A4B/A4A identities, source/checkpoint/reset hashes, exact inherited criterion, ED/increments/thresholds, H_eff/processed scores/consensus/ties/criterion, label and reference barrier/secondary flag. Share locked metric primitives only.

Allowed exposed D-dev-select PATH0 exact12 and sealed A4/PATH0 artifacts. Forbidden fresh validation, D-dev-confirm/D-test/router-calib new role, full24 extension/full100 PATH1/full300/P3, SEAME/CS-FLEURS/ViMedCSS/ASCEND/transfer; no steering or new TTA settings. No outcome report now. After later audit/report, STOP regardless of result; no deployment or follow-on authorized. All reviewed changes: checks→diff→commit→push origin HEAD:cs-asr-steer-inf; never force/main merge/PR.
