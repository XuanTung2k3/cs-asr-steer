# A5000 E0 scientific setup — new pilot v1

This is a new training-partition pilot, not authenticated replication of missing H100 P2 code/results.
Proposal reviewed: local ReFT Research Proposal Workflow 2026-10-19, updated 2026-10-20, heading v5;
SHA256 ff580a9f318560f2db760ca3c3b6c3b903d2c136e0638a2a3467e717d665b711. No v5.1 found.
One bounded inventory of the known local stage, ACL draft, Downloads, remote tung.nx/shared research roots
found no newer encoder source, P2-F1 or train394/inner106/full500. No H100 access.

## Roles, frozen before model-quality inspection

Only the verified official TRAIN manifest is used (release a84770fcdf14072252373a0227321227eaac7dc7).
Eligible duration 1–30 seconds; <=190 content tokens under the pinned tokenizer; no audio cropping.
Exclude OTHER strata and all duplicate prefix-hash members. Seed42 permutation of 70 dialogue IDs allocates
42/14/14 whole dialogues. Seeded within-dialogue order and stratum round robin select exact quotas.
All 650 selected files have full SHA256; identity, audio, speaker and dialogue disjointness is checked.
Exact target-text normalization is reused unchanged from reft.text.target_text; target hashes are recorded.

| Role | n | Dialogues | Speakers | CS / ZH-only / EN-only | Substantive EN units | Audio seconds |
|---|---:|---:|---:|---|---:|---:|
| A5000_T394 | 394 | 42 | 84 | 276 / 79 / 39 | 2497 | 4942.309 |
| A5000_V106 | 106 | 14 | 28 | 74 / 21 / 11 | 663 | 1403.630 |
| A5000_C150 | 150 | 14 | 28 | 105 / 30 / 15 | 939 | 2053.708 |

Versioned IDs/metadata: configs/reft/a5000_manifests/roles_v1.json.
Manifest SHA256: ad43487483ab70fb717590cf1e7e805b771dce822cfad3bde4db7746f5c8eb2b.
The ID manifest contains no transcripts/audio; original private training metadata remains outside Git.
T394 fits adapters; V106 selects LR/checkpoint; C150 is reserved and blocked by the initial campaign reader.
No official dev/test transcript/audio is used by this experiment. New roles do not recover historical identities.

## O1 pilot protocol, predeclared

Exactly two primary arms: encoder-output E-ReFT and same-site LRA, rank2, 5122 learned scalars, seed42.
Frozen BF16 large-v3; FP32 adapters; post-final-encoder-layer-normalization site, all 1500 frames including padding.
Both use AdamW betas(.9,.99), eps1e-6, weight_decay0, clip1, constant LR; model eval disables dropout/SpecAugment.
Two equally budgeted LR candidates per method: 1e-4 and 5e-4. Each trains one full epoch. Choose candidate by
V106 (genuine-CS MER, overall MER, ZH-CER, candidate order). Continue only that candidate through epoch3.
Select its checkpoint among epochs1–3 by (genuine-CS MER, overall MER, ZH-CER, earliest epoch).
No claim that these LRs are optimal or authenticated historical encoder recipes. No candidate expansion.

Microbatch1, effective batch16; final partial batch retained. CE is total content+EOS NLL divided by TOTAL
supervised tokens in the logical batch. Each microbatch contributes its NLL sum / that common denominator.
This is an explicitly new reduction policy, not assumed equivalent to historical microbatch-mean accumulation.
Teacher forcing: prompt [sot,frozen detected language,transcribe,notimestamps], logit 3+t predicts target[t].
Prompt positions0..2 and padding ignored; EOS once. Pure CE: GoldKL=0, LID=0, weights=1. No augmentation.

Frozen Auto-LID is computed from unadapted encoder states before intervention. Generation uses that detected
language, beam5, deterministic, max_new_tokens200, no previous-token conditioning; no gold inference language.
Both arms use exactly the same data, cache, decoder and canonical scorer. Nontermination/loops remain in metrics.
Checkpoint every optimizer boundary, immutable epoch snapshots, resumable per-utterance validation outputs.
At most one GPU and six aggregate resident GPU-hours, including failed/preflight runs. No C150 selection.

Unknown historical details: exact encoder source/init mechanics, padded-frame convention, optimizer/reduction,
GoldKL recipe, role membership and approvals. We reuse verified published orthogonal parameterization and
CE/text/scoring primitives, and state all new choices above rather than implying historical equivalence.

## Conditional post-selection confirmation (declared before C150 model evaluation)

If both primary arms finish within the resource envelope with at least 1800 seconds of GPU budget remaining,
run one paired confirmation event: frozen B0 plus each method's ALREADY selected LR/checkpoint on C150.
No retraining, checkpoint/LR changes or C150 model selection. A separate immutable confirmation freeze binds
selection SHA, both checkpoint hashes, source commit and role manifest before any C150 model forward.
This is an original-TRAIN development holdout, not an official/external test. After this event C150 is exposed
for these hypotheses and cannot be relabeled fresh for later objective studies. The primary campaign itself
retains its C150 firewall. Conditional confirmation uses a separate guarded reader and shares the six-hour cap.
