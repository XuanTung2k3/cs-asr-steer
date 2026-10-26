# RC7-0B — Data roles and holdout-exposure audit

Method: metadata only. No C500 audio, transcript, prediction or score was read, decoded or computed. Tools:
`scripts/rc7/rc7_0_audit.py exposure|history` (outputs in `evidence/`; counts and hashes only, no IDs).

## 1. Status flags

| Flag | Value |
|---|---|
| C500 model-level exposure within A5000 (any adapter fitted, selected, decoded or scored on C500) | **NONE FOUND** |
| C500 historical (steering-era) exposure | **`C500_PARTIAL_HISTORICAL_EXPOSURE_VERIFIED`** — 41 / 500 utterances (8.2 %), 10 / 14 dialogues |
| C500 mapping to historical H100 ReFT roles (Train394/Inner106/Full500/Dev150) | **`C500_H100_ROLE_MAPPING_UNRESOLVED`** — manifests missing |
| Official test split (dialogue-v2 `D-test` = 15 official-test dialogues) vs A5000 roles | **disjoint by partition** — every A5000 row has `official_split = train`; no A5000 run artifact names a test file (the downloaded BAAI/CS-Dialogue rev `a84770fc…` archive on disk contains all splits) |
| C500 eligibility for "untouched / independent held-out" claims | **NOT ELIGIBLE as stated**; eligible only as a never-A5000-evaluated TRAIN-derived development holdout with disclosed partial exposure, subject to an owner decision (§6) |

## 2. Split regimes (not interchangeable)

| Regime | Roles | Provenance |
|---|---|---|
| Steering project, dialogue-v2 (`configs/lss/roles.yaml`, `docs/current/DATA_EXPOSURE.md`) | D-construct 20, loc-train 15, util-train 12, router-calib 8, D-dev-select 20, D-dev-confirm 10 dialogues (pooled official train+dev); D-test = 15 official-test dialogues | dialogue-atomic rerandomization; role tree on H100 storage |
| H100 ReFT (decoder site) | ACL-0 dev150 (⊂ D-dev-select, exposed, used for selection); training from loc-train ∪ util-train; "Train394/Inner106/Full500/Dev150" | **exact manifests MISSING** |
| A5000 E1 (`roles_v1.json`, sha `ad434874…`) | A5000_T394 / V106 / C150 | seed-42 permutation of 70 official-train dialogues, 42/14/14; *new identities*, `historical_replication: false` |
| A5000 campaign (`data_manifest.json`, sha `fc170e10…`) | A5000_T2000 / V300 / C500 | same 42/14/14 dialogue allocation, extended; membership sha256 T `14e032a4…`, V `88a34b86…`, C `3ab45e73…` |

Names ending in "500" or "150" do not identify the same population across regimes: A5000 C150 is not H100 Dev150,
and A5000 C500 is not H100 Full500.

## 3. A5000 role integrity (verified)

- Counts T2000 2000, V300 300, C500 500; C150 ⊂ C500 (150/150); T394 ⊂ T2000; V106 ⊂ V300.
- Pairwise utterance-ID and audio-sha256 overlap between T/V/C roles: **0** in every pair.
- C500 membership sha256 recomputed = frozen `role_identities.json` value `3ab45e73…`.

## 4. A5000-internal exposure of C500/C150 (none found)

Scan of `a5000_experiments/`, `a5000-e1-pilot-v1/`, `a5000-validation/`, `a5000-bootstrap/` (21,319 text files plus
all filenames):

- No encoder-cache, prediction or any other file named after a C500/C150 utterance. Core encoder cache holds
  exactly 2,300 entries (= T2000 + V300).
- No `confirmation/` directory or `confirmation_freeze.json` in core7h-20261020-v2; N5 never started.
- The only file mentioning holdout IDs is the E1 pilot `audits/data_and_target_integrity.json` (150 mentions):
  a waveform-hash duplicate check over all 650 E1 rows with `C150_model_evaluated_by_this_audit: false` and
  target sequences verified for 500 rows (T394+V106 only). This is administrative audio-identity access, not
  outcome exposure. E1 `comparison.json` records `C150_evaluated: false`.
- The sprint `data_manifest.json` lists C500 identifiers for integrity inventory; sprint code filters to T/V before
  materializing references (`docs/reft/a5000_lra_sprint/PROTOCOL.md`).
- Limitation: file-access times are not a reliable record (relatime); this is absence-of-artifact evidence plus
  code-path firewalls, not an access log.

## 5. Historical steering-era exposure of C500 (verified)

The steering project's tracked artifacts (committed 2026-09-04, `97d7f04`) contain utterances that the later A5000
sampler placed in C500. The A5000 sampler ignored steering roles, so this overlap was not prevented.

| Steering-era artifact (role per `DATA_EXPOSURE.md`) | Use | C500 utterances | C150 | C500 dialogues |
|---|---|---:|---:|---:|
| `results/job_a/F0–F5_hypotheses.jsonl` (D-dev-confirm subset) | decoded and scored vs references (frozen + steered Whisper) | 17 (all CS) | 6 | 3 |
| `results/round1/job_a/baseline_beam5.json`, `out/_determinism_probe_*` (D-dev-select subset) | decoded and scored | 13 (all CS) | 3 | 2 |
| `results/job_b/train_utterance_ids.json` (loc/util-train sample) | training list for steering gate/direction | 11 (9 CS, 1 ZH, 1 EN) | 3 | 5 |
| **Union** | | **41 (8.2 %)** | **12** | **10 of 14** |

Dialogue-level inference (via conversation IDs; no dialogue maps to more than one steering source, consistent with
dialogue-atomic roles): C500's 14 dialogues = 3 steering D-dev-confirm, 2 D-dev-select, 5 loc/util-train,
4 unobserved. By symmetry, A5000 T2000 (training) contains 5 steering D-dev-confirm and 9 D-dev-select dialogues,
which affects the steering project's confirmation role, not RC7.

What this does and does not imply:
- No A5000/ReFT adapter was trained, selected or evaluated on C500.
- Researcher-level outcome exposure exists for 30 C500 utterances (frozen/steered Whisper transcripts and errors),
  and 11 were inputs to a different method's training. Two C500 dialogues lie in the steering D-dev-select role,
  of which H100 ACL-0 dev150 (used to make H100 ReFT design choices) is a subset; whether they intersect dev150
  itself is unresolved.
- C500 therefore cannot be called untouched or fully independent.

## 6. Eligibility and required owner decision (no new split created)

RC7-0 does not redefine roles or create a substitute population. Before any RC7-5 held-out claim, the owner must
choose and freeze, *before opening C500*, one of:

1. Use C500 as a TRAIN-derived development holdout with the exposure above disclosed, plus a predeclared
   sensitivity analysis excluding the 41 exposed utterances (and, separately, the 10 exposed dialogues), or
2. Restrict independent-evaluation claims to a different, already-defined locked population (e.g. the official
   test, which A5000 never touched), under its own lock rules, or
3. Recover the steering dialogue-v2 role tree and H100 ReFT manifests
   (`/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles`, H100 `assignment.json`) to resolve
   §5's open intersection before deciding.

The exposed-utterance list itself is not published here. It is reproducible from tracked repository files with
`scripts/rc7/rc7_0_audit.py history`.

## 7. Development-selection exposure of the other roles

| Role | Exposure |
|---|---|
| A5000_T2000 | training for all adapters and gates (gate normalization fitted on T only) |
| A5000_V300 | **all** LR/epoch/alpha/gate selection and every reported development number; heavily exposed; descriptive only |
| A5000_V300[:64], T2000[:64] | N4 D1/D2 diagnostic panels (predeclared; diagnostic, not claims) |
| H100 dev150 | exposed; selected every H100 checkpoint |
| Official test / D-test | reference-only counts and frozen C00 baseline used for power analysis (steering era); never intervened; untouched by A5000 |
