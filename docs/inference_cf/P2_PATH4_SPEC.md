# Active revision: PATH4_R1_EOS_BOUNDARY_V1

Human-authorized EOS-boundary scope extension: [P2_PATH4_EOS_AMENDMENT.md](P2_PATH4_EOS_AMENDMENT.md).
This amendment supersedes ONLY revision0 EOS-first invalidity and its pre-run block.
CONTENT_G1 and all original scientific settings/thresholds/firewall remain unchanged.
New contract gate: PASS_TO_P2_PATH4_R1; runnable manifest/code must also be independently audited before GPU.
No scientific run/outcome/reference evaluation in this Codex session.

The complete blocked revision0 below is preserved verbatim as historical provenance, not current execution status.

---

# P2-PATH4 — Frozen G1 Transfer to A2 on Fixed100

**Status: BLOCKED BEFORE EXECUTION — not ready for Claude scientific execution.**
Starting local and remote HEAD: `53a6a62a90a596c65f3bd2dea5f57d0410ef401d`, local branch
`feature/inference-cf-steering`, clean tree, empty user queue. This is a separate exposed-development
inference_cf contract, not a revision of the core v6 METHOD_CONTRACT. Historical stages remain closed.
No runner, adaptation, model inference, scientific outcome, or GPU job was produced in this freeze.

## Discovered pre-run incompatibility: STOP

The requested unchanged G1 transfer cannot execute on the complete fixed100. From sealed reference-free
B0/A2 content IDs and EOS termination, four first disagreements are **theta0 EOS versus A2 content**:

| Utterance | Zero-based content decision k | theta0 | A2 | Existing partition |
|---|---:|---:|---:|---|
| ZH-CN_U0027_S0_116 | 75 | 50257 (EOS) | 26636 | DEV24 |
| ZH-CN_U0086_S0_222 | 35 | 50257 (EOS) | 26636 | NOVEL76 |
| ZH-CN_U0091_S0_196 | 84 | 50257 (EOS) | 11 | NOVEL76 |
| ZH-CN_U1004_S0_221 | 50 | 50257 (EOS) | 76 | NOVEL76 |

Each B0 is a strict content-prefix of A2 and ends with EOS. Both models must therefore disagree at k
if their historical outputs reproduce. The theta0 H=3 rollout has zero content tokens. The frozen
PATH2 spec explicitly makes EOS-first branches INVALID; PATH3 retained that rule. Actual
`inference_cf_p2path3.online_g1` raises `TTAInvalid` if either H_eff < 1, and
`branch_adjudication.score_branch` rejects empty donors and any EOS donor.

This is an implementation **scope incompatibility**, not a numerical mismatch or new scientific outcome.
No `PASS_TO_P2_PATH4` can be issued under this exact contract. STOP before adaptation/GPU execution.
Do not exclude these rows, ignore EOS disagreements, invent an EOS score, copy A2 through, or modify
G1 to make the job pass. A separately authorized revision is required. The remainder records the
requested unchanged transfer and frozen analysis, so the blocker and scientific intent are reviewable.
It does not authorize that revision or any scientific run.

## Scientific question and fixed panel

Can unchanged PATH2 CONSENSUS-1, replacing only the adapted actor A4 with A2, retain A2's broader
English/POI benefit while reducing Mandarin path damage? PATH3 concluded SIGNAL_CONCENTRATED: A4
was inert on NOVEL76. Controller tuning is excluded.

Use the exact historical TTA1 panel: 100 utterances, 20 dialogues × 5, parent byte SHA256
`266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`.
Parent internal panel hash has a different domain; never confuse it with the byte hash.
`P2_PATH4_PANEL.json` fingerprints canonical order, IDs/dialogues, parent rows, audio bytes, teachers,
B0/AUTO/A2 outputs and all100 final-master archives. Partition membership is output-token/ID-only:
FULL100=100; DEV24=24; PATH2_12=12; NOVEL76=76; A2_DELTA=14; A2_SAME=86;
A2_DELTA_OUTSIDE_DEV24=7. PATH2_12 ⊂ DEV24 ⊂ FULL100. DEV24/PATH2 retain native parent order;
new partitions filter canonical FULL100 order. Membership hashes serialize compact sorted-key UTF8
JSON `{partition,rows:[{utterance_id,dialogue_id}]}`. No reselection or new data role.
NOVEL76 is already TTA1-exposed development, not fresh validation.

## Exact historical A2

Authority: frozen TTA1 plan/selection, original `episodic_tta.adapt(kind="A2")` and TTA1 runner.
Teacher is sealed theta0 AUTO greedy content IDs y_A; student is forced-ZH, with prompt
`[50258,50260,50360,50364]`. No new AUTO teacher generation, language detection, soft teacher or A4 no-op rule.
Loss is mean valid-position `-log p_theta(y_A[t] | x,cB,y_A[:t])`. Valid masks are original sealed TTA1
masks; input cB+y_A, no appended EOS, query index3+t, use_cache=False, no past KV. Prefix/EOS/special
positions do not contribute loss. EOS remains in the allowed predictive vocabulary. General suppression
applies at every absolute content position; begin suppression only at t=0. Float32 log_softmax over
allowed IDs, temperature1. Preserve raw IDs, never retokenize text. Empty valid set gives exactly two
connected-zero no-op optimizer steps; never drop/refill a row.

Trainables are exactly the config-enumerated 194 decoder LayerNorm affine tensors, 248,320 scalars:
32 layers × self_attn_layer_norm/encoder_attn_layer_norm/final_layer_norm × weight/bias, plus final
model.decoder.layer_norm weight/bias. All resident model parameters require_grad=False; only fresh
fp32 masters require gradients. AdamW lr1e-3, wd0, betas(.9,.999), eps1e-8, amsgrad/foreach/fused/maximize
False, exactly2 updates. No scheduler, clipping, scaler, autocast or accumulation. bf16/eager eval
functional_call forward with masters cast bf16; seed240924. Exact episodic reset and immutable non-LN
parameters. Frozen detached encoder computed once per row; clone outside inference_mode for adaptation.
Final ordinary forced-ZH decode: greedy, batch1, beam1, no sampling, max200 output decisions including
EOS, passive L16 recorder and attention=True cached.Branch execution, no action hook/steering/D2.
Exact suppression arrays and complete parameter enumeration are in the config.

## Reconstruction and unchanged transferred G1

If a future authorized contract resolves the known scope block, require a two-phase barrier:
first reconstruct/free-decode A2 for **all100**, comparing historical tokens and termination exactly,
all194 fp32 final-master arrays and their bf16 effective values exactly. Array equality is required;
new NPZ container byte equality is not, but historical NPZ bytes must match pinned hashes. Require
matched theta0 B0 tokens/termination too. Any genuine mismatch is INVALID; never alter A2 to match.
Keep tiny per-row states/encoder outputs, then materialize each saved A2 for controller execution;
no second adaptation. Destroy caches and reset after every row.

G1 inherits unchanged PATH2/PATH3 primitives. Two independent resident theta0/A2 model instances,
identical frozen non-LN weights, clean model-owned decoder paths. Before intervention compare
processed greedy argmax at the live common prefix. First disagreement triggers once; equal EOS
terminates, cap200 terminates. Expected trigger set **equals all14 A2_DELTA**, including four EOS
mismatches; expected k/prefix/first IDs are audit-only metadata, never runtime lookup features.

At a supported trigger, live owner-specific greedy b0/b2 rollouts have H=3 content tokens, shortened
at EOS/cap, at least1 each. First tokens differ. No AUTO/donor branch. Score each under both models
with unchanged fresh-cache PATH1 scorer, mean float32 processed token log probability, full allowed
vocabulary, absolute indices, temperature1, EOS excluded. Equal-weight consensus:
`S_cons(b)=.5*S_theta0(b)+.5*S_A2(b)`. Exact tie implementation: theta0 if
`S_cons(b0)-S_cons(b2)>=-1e-12`, otherwise A2. Emit only winner's first token, consume under A2 and
immediately resume A2; never guard again. No theta0 handoff/G2 or stale cross-model KV.
NO_TRIGGER and A2-winner outputs must equal ordinary A2 token/termination exactly.
Legacy helper names A4/b4 may remain as an explicitly documented adapted-role alias for A2/b2;
actual objective/state hashes must identify A2. This is metadata plumbing, not A4 adaptation.

## Reference-free seal and metrics

Reuse hashed historical B0-FORCED, B0-AUTO and A2. Only A2+G1 is a new method; reconstructed A2 is
integrity-only. PATH3 A4+G1 may be displayed from sealed outcomes after the reference barrier; never rerun A4.
Runtime controller inputs exclude historical reference/error/POI labels, per-row A2 success/failure,
partitions, sites and donor outputs. Log IDs/dialogues/partition metadata separately from runtime,
reconstructed tokens/termination/matches, trigger/k/prefix hash, b0/b2/H_eff, four score paths,
consensus/winner/margin, G tokens/termination/change, token ED to A2/B0/AUTO, EOS/length/cap,
state/cache/reset evidence, update/call counts, runtime and peak allocated/reserved VRAM.

Before references summarize FULL100/DEV24/NOVEL76/A2_DELTA_OUTSIDE_DEV24 triggers/no-triggers,
winners/changes/trigger dialogues and triggered signed theta0-minus-A2 margin quantiles0,.25,.5,.75,1
(null if no triggers). Seal all100 outputs/logs, partitions, manifests and source/config hashes;
commit and PUSH immutable seal; independent primary audit PASS; only then open references/evaluator.
Incomplete outputs never authorize evaluation. No observation may change the policy or thresholds.

After seal, canonical evaluator reports PIER/MER/EN-WER/ZH-CER, ZH/POI/mixed error counts,
POI corrections/corruptions, matrix-ZH/embedded-EN retention, outside-POI lexical harm, S/D/I,
caps/severe truncations. Main FULL100 four-system table plus partition opportunity/outcome summaries.
AUTO is practical comparator, B0 is causal baseline. Report theta0-winner improving/worsening ZH/POI/mixed
rows descriptively only. No new reference evaluation or row-level historical outcome analysis was done here.

## Frozen outcome thresholds

Authoritative TTA1 safety bounds against B0 all apply: ΔMER<=.01, ΔZH-CER<=.015,
ZH retention>=.98, EN retention>=.95, outside harm<=.03, POI corruption<=.05,
added caps<=1, new severe truncations=0. Float guard1e-12. Preserve canonical denominator semantics:
primary POI/EN/ZH denominators nonempty; empty baseline-correct rates=None, explicitly not-assessable
and vacuous, never invented0/1. Severe truncation: B0 content length>=10, new EOS termination and
new length<=floor(.5*B0 length). All eight pass → ABSOLUTE_SAFETY_PASS.

Already-exposed aggregate tables independently agree: B0 ZH1107/POI345/mixed1483; A2
ZH1116/POI316/mixed1464; POI denominator696. Post-seal evaluator must reproduce these historical
aggregate baselines exactly or INVALID. No row-level outcomes chose the partitions.

AGGREGATE_RESCUE_PASS: `X=max(Z_A2-Z_B0,0)`, required=`max(1,ceil(.50*X))` if X>0 else1;
`R=Z_A2-Z_G >= required`. Verified X=9, required5, hence **Z_G<=1111**.

BENEFIT_RETENTION_PASS: `I_A2=345-316=29`, `I_G=345-POI_G>0`, `I_G/29>=.75`,
PIER_G−PIER_A2<=.01, MER_G−MER_A2<=.005, and **Mixed_G<=1464**. Mixed count is a stable
canonical count and a required supportive quality guard, not merely descriptive.
Retention alone requires at least22 net POI errors saved (POI_G<=323); the .01 PIER guard with
fixed denominator696 permits at most6 extra POI errors, so the combined POI cutoff is **322**.

BREADTH_PASS uses FULL100 r_i=ZH_A2(i)−ZH_G(i), p_i=max(r_i,0), h_i=max(-r_i,0).
Report positive rescue, new harm, net rescue; P_d=sum p_i in dialogue. Require all six:
≥3 positive-rescue utterances; ≥3 positive-rescue dialogues; max_i p_i/sum p_i<=.50;
max_d P_d/sum p_i<=.60; net rescue on NOVEL76>=1; ≥1 positive-rescue NOVEL76 utterance.
If positive-rescue denominator=0, both concentration shares=1. Do not inherit PATH3's NOVEL-only
three-utterance criterion or count old rescue as outside DEV24.

LODO is descriptive: FULL100 net rescue after removing each of20 dialogues, and after removing each
controller-changed dialogue (defined by sealed G!=A2 token/termination). Min/median/max/count/proportion>0;
empty changed set gives null summaries/count0. No label override.
Bootstrap is descriptive paired dialogue blocks: B2000, seed240924, numpy default_rng/PCG64,
sorted20 dialogue IDs, sample20 with replacement each draw, same draws for systems, sum canonical
counts then rates, percentile95% intervals. G−A2 and G−B0 PIER/MER/ZH-CER/POI error counts;
report omitted zero-denominator draws, no significance gate.

## Terminal precedence and audit

First applicable, without post-outcome modification:
1. P2_PATH4_INVALID: any known pre-run incompatibility, provenance/reconstruction/trigger/cache/controller/
   reference/evaluator/budget/audit failure. **The current known EOS-first block stops here.**
2. P2_PATH4_SEQUENCE_DAMAGE: absolute safety fails.
3. P2_PATH4_SAFE_NO_ADDED_VALUE: aggregate rescue fails.
4. P2_PATH4_OVERCONSERVATIVE: benefit retention fails.
5. P2_PATH4_SIGNAL_CONCENTRATED: breadth fails.
6. P2_PATH4_GUARD_TRANSFER_SUPPORTED: all gates plus independent post-audit PASS.
Supported would justify only separately frozen independent confirmation, never full300/P3 here.
No outcome label is claimed in this design session; the known pre-run block is not a run report.

Independent auditor must not import primary decision code. Pre-run gate PASS_TO_P2_PATH4 must
verify hashes/partitions, EOS candidate eligibility, A2 inheritance and100 checkpoint targets,
G1 source identity, trigger relation, owner/cache isolation, score/tie/commit semantics, all thresholds,
reference firewall and compute. Current four blockers require refusal of that gate. Later primary/post
checks independently reconstruct controller arithmetic, identities, canonical counts/safety/rescue/retention/
breadth/bootstrap/labels and require P2_PATH4_AUDIT: PASS. No continuation on failed audit.

## Tests, compute, firewall and GitHub

Focused synthetic/CPU contract tests: exact100/20/order/hash/partitions, token-only delta14/same86/outside7,
four EOS-first blockers, historical source/teacher/checkpoint hashes, A2 parameter list/optimizer/loss/mask/reset,
matched B0/A2 path, first-disagreement equality including EOS, H3/weights/tie/one-token/one-event inheritance,
model-owned KV, no-trigger/A2-winner identity, no alternate EOS fallback, zero steering/D2/G2/A4 calls,
all threshold boundary formulas/label truth table, seal-before-reference and independent audit separation.

Current authorized scientific jobs=0. Requested budget, usable only after an explicitly revised contract:
at most1 sbatch MIG/H100 job, canonical100,200 updates, at most201 backward calls including one independent
live audit (loss tolerance1e-5, gradient relative.02/floor1e-8), target<15min, hard30min, no expansion/resubmission.
No A4/A3/G2/hyperparameter/controller sweep, full300/P3 or confirmation.
Allowed already-exposed D-dev-select fixed100 and sealed B0/AUTO/A2/A3/A4/PATH artifacts only.
Forbidden fresh validation/D-dev-confirm/D-test/router-calib new role/new panel/split/full300/P3/SEAME/
CS-FLEURS/ViMedCSS/ASCEND/transfer. Checks→diff→commit→push origin HEAD:cs-asr-steer-inf;
clean local=remote; no force-push/merge-main/PR. Historical reports and core contract are preserved.
