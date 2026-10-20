# A5000 energy-allocation campaign v1

This new TRAIN-only campaign follows the owner's N0–N5 request. It does not amend historical P2/P3 freezes.
Source is isolated on exp/a5000-energy-campaign-v1. ReFT/LRA reuse the A5000 reconstruction, not missing H100 code.

## Data and exposure

New T2000/V300/C500: preserve the previous pilot's 42/14/14 dialogue allocation. The earlier V106 is already
exposed development; its larger V300 role remains selection-only. The C150 dialogue allocation must have no
previous confirmation predictions. New C500 is one fixed confirmation package after final freeze, never a
selection panel. All rows belong to original CS-Dialogue TRAIN. No official dev/test files are opened.
Sampling is seed42 per-stratum dialogue round-robin (70% CS/20% ZH-only/10% EN-only). Canonical eligibility
is 1–30 s and <=190 content targets; all exclusions are inventoried. Both full-file and decoded-waveform
hashes are checked; identity, dialogue, speaker and waveform roles must be disjoint. No crops/beam changes.

## Objectives and methods

N1: B0, post-final-encoder-LN rank2 ReFT/LRA (5122 each), last-encoder q/v LoRA r1 (5120), all-encoder
q/v LoRA r8 alpha16 (1,310,720). Strong means broad practical capacity, not proven best-tuned SOTA.
All backbone parameters frozen. BF16 backbone, FP32 adapters. Valid encoder frames only in this version;
therefore earlier all1500-frame pilot checkpoints are not reused as matched baselines.
All arms use AdamW(.9,.99), eps1e-6, weight_decay0, clip1, effective batch16/microbatch1, dropout/augmentation
disabled, constant LR. Each method has two full three-epoch trials (1e-4,5e-4), equal candidate/horizon budgets.
This intentionally uses full-horizon LR selection instead of the older pilot's one-epoch trial continuation.
Select by genuine-CS MER, overall MER, ZH-CER, earliest epoch. No C500 selection.
Frozen-backbone Auto-LID first, then adapted beam5/max200 deterministic generation. Targets use prompt plus
content[:-1], content+EOS supervision at predictive indices3+t; prefix0–2 ignored. Logical-batch token-global CE.
Encoder weight-LoRA recomputes the encoder and checkpoints frozen-mode layers; output caches are forbidden.

N2: CE warm-start from selected R; gate289 parameters (16 inputs,16 hidden,1 output; total5411).
One-second blocks at measured .02s stride =>50 frames. Projected means/variances, log RMS update, state/update
cosine, neighboring contrasts, state RMS, valid fraction. Z-score fit on T2000 only. No transcript gate inputs.
Generic controls share MLP/feature capacity and acoustic feature access; update features masked. Acoustic and
geometry ablations mask inputs without reducing capacity. Initial final MLP layer zero =>exact warm identity.
Epsilon .25, alpha1. Coefficients conserve proposed FP32 valid-frame update energy per utterance; double-precision
ratio, exact zero-energy scalar fallback, no coefficient clipping. BF16 cast drift reported separately.
G3 unconstrained, G4 generic energy-normalized, G5 proposed energy-normalized. G8 warm continuation sees one
extra T2000 epoch exactly as gates. Global alpha candidates[0,.5,1,1.5,2]. Gate LR.001, one epoch, adapter frozen.
No GoldKL, no LID loss, no language/POI labels in N1/N2.

Frozen exploratory GO: CS-MER improves >=.005 vs G1,G3,G4,G8; no additional loops/nontermination; non-CS MER
worsening <=.01; >=100 CS and60 nonCS utterances; positive error-count gains span>=3 dialogues with no single
dialogue>50% of positive gain; allocation SD>=.001. Missing support =>INCONCLUSIVE. PIER unavailable absent
verified POIs, never manufactured from scripts. No threshold/architecture search after outcomes.
GO opens G2/G6/G7, energy-shuffled G9 and gated LRA. NO-GO/inconclusive do not open those arms but N3/N4 continue.

N3 reuses N1 CE and runs same two LR/three-epoch budgets with normalized WCE, frozen singleton omega2.
Weight2 is assigned only to safely decoded Latin-script target tokens inside genuine-CS training utterances;
ambiguous/control/EOS and EN-only targets weight1. This is an explicit English-script embedded-token proxy,
not authenticated linguistic annotation. Sum weighted NLL/sum weights over logical batch. Audit reports coverage.
GO also opens diagnostic G5 WCE from WCE warm start; it is a different recipe, not a pure gate effect.

N4 includes bounded D1/D2 on first64 T and64 V IDs (deterministic IDs). Probes are fit on T only, evaluated V;
fixed/refitted/prefix-zero-memory controls. Script groups are declared proxies. These are not causal evidence.
D3 has no sufficiently specified matched-random recipe and remains BLOCKED in v1, not fabricated.

N5: seeds42/43/44 for decisive R/L/LS CE, with seed42-selected LR/epoch horizon frozen. Seeds43/44 train fresh
adapters; eligible gates warm-start their own independent adapters. Final epochs fixed, no per-seed selection.
Freeze exact candidate/checkpoint package before C500 opens. Same package can resume per-utterance; additions
or retuning forbidden. Weighted cells remain one-seed exploratory under resource priority. C500 remains TRAIN-derived
development confirmation. Independent official/external evaluation BLOCKED pending separate scoped authorization.

## Operational limits

One idle GPU, 72 allocated GPU-hours INCLUDING preflight/failures,120h wall,36h stage limit,20min preflight,
150GiB output cap,30GiB free-space floor,92% GPU reserve threshold,3h reporting/measurement reserve.
Actual worst-envelope forward/backward and beam5 throughput must admit N1+N2 within cap before launch.
No automatic recipe-changing OOM retry. Controller/child interruption replays incomplete logical batch from
optimizer/RNG/cursor checkpoint. Constant scheduler/no scaler are explicit. No random/dropout data transforms.
Exclusive campaign and GPU locks; tmux survives disconnect. Reboot recovery is explicit resume, not automatic.
One controller-infrastructure restart is allowed; a surviving child must finish before replacement. Scientific/numerical failures do not change recipes or automatically retry. Explicit resume recovers compatible checkpoints.

## Claim limits

Internal Codex review, not external audit. Energy identity is not a novelty proof. Per-arm training costs are
not full pipeline efficiency. No superiority conclusion from single-seed selection, engineering PASS, or probes.
PIER/Chinese-word WER are unavailable absent frozen valid annotations/segmentation. All insertion/loop outputs
remain in primary MER. Reporting must expose missing analyses/contrast/resource truncations.

E-ReFT and same-site affine LRA belong to closely related low-rank affine function families; no novelty or ReFT-specific superiority follows from their formulas. Internal code-path inspection checked post-LN placement, CE-only path, audio-only gate inputs, valid-frame energy, warm identity, frozen backbone, matched continuation and C500 firewall. This is an internal Codex self-review, not an external audit.
