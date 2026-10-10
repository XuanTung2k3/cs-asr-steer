# DIR-SPRINT0 reference firewall and independent audits

The freeze is metadata, source and CPU-math only. Historical aggregate reports motivate the design; no new lexical
outcome or reference transcript is inspected before the PRIMARY audit.

## 1. Input separation

**Population and bank selection.** The 240 prospective utterances and the 300-utterance calibration bank are selected
from:
- the six-column role metadata;
- the hash-checked exposure registry (`exposure_registry.exposed_ids()`, ledger + append-only addenda).

No reference, label, baseline output, phonetic score or gate value participates.

**GPU runner input.** The runner reads only `runtime_panel.json`:
- `rows` with exactly `canonical_index, utterance_id, audio_path, audio_full_sha256`;
- `bank_rows` with exactly `bank_index, utterance_id, audio_path, audio_full_sha256`.

Extra or missing fields are rejected recursively. GPU phases never read the population file, dialogue IDs or roster
metadata (statically tested).

**Construct stage (CPU, reference-free).** It reads dialogue IDs from the frozen population for three purposes only:
- the original D1 fold assignment (fold d for dialogue d);
- D5 bank dialogue balancing;
- cluster bookkeeping.

Dialogue identity is never a lexical label.

**Direction inputs.**
- **D2:** its own script-mass objective.
- **D3:** B0 and null-audio logits of the same query, and its own token-margin objective on the runner-chosen c_AP.
- **D4:** transcribe and translate states.
- **D5:** phone posteriors of the original audio, the frozen tables and unlabeled bank states.

None receives a reference, acceptable-token set, competitor derived from references, stratum, gold phone or timing,
language label, CTC/MMS-FA timing, oracle window or evaluator gradient.

**Phone provider.**
- The adapter accepts audio only. phonemizer and espeak are absent, and the provider tokenizer class is unused.
- Engineering clips are public (LibriSpeech, AISHELL-1); no CS-Dialogue reference is involved.

**Model weights** never change. No optimizer is used, and there are no parameter gradients.

## 2. Barriers, in order

1. Commit and push the design freeze (spec, design, firewall, compute, population, config, concept and engineering
   artifacts, freeze index, contract tests). Then commit and push the implementation, runtime panel and manifest.
2. Independent `PASS_TO_DIR_SPRINT0`, which covers:
   - pins, population recomputation and exposure exclusion;
   - the D3 legal set;
   - the concept table recomputed from panphon;
   - provider and table digests;
   - D1 folds;
   - static firewall and semantics, Slurm contract, storage, Slurm queue, and tests.
3. Job A: capture, phone children and construct. No pulse, readout autograd or reference.
4. Commit and push `job_A_seal.json` (every row and array hash, content-addressed archive). Then the independent
   `DIR_SPRINT0_AUDIT_A: PASS`, which covers:
   - gate and compatibility recomputation;
   - D3, D0, D1, D4, RND, v_AC, D5 evidence and prototypes, D5 and D5SH recomputation;
   - provider spot recomputation;
   - opened-path allowlists.
5. Planned-pulse recost (`resource_forecast.json`) and the minimal eight-field `authorization_B.json`, which carries no
   IDs, labels or per-row outcomes.
6. Job B pulses. No references.
7. Commit and push `pulse_seal.json`. Then the independent `DIR_SPRINT0_AUDIT_PRIMARY: PASS`, before any reference is
   opened.
8. The separate evaluator filters the role parquet to the 240 frozen IDs before materializing `transcript_raw`, maps
   every query and counts every frozen metric. `evaluation_units.json` (reference surfaces) is kept out of Git; its hash
   is in `evaluation.json`.
9. Independent `DIR_SPRINT0_AUDIT_FULL: PASS`: its own mapping (`srd2_g0_audit.own_map`), counts, bootstrap,
   predicates, family statuses and terminal label.

No stage continues on a failed audit, a missing remote seal or a changed pin. Failed attempts and opened-path logs are
preserved. There is no automatic job rerun, population reduction or refill.

## 3. Independent auditor restrictions

**Imports.** The auditor never imports:
- `experiments.inference_cf_dir_sprint0` or `csasr.inference_cf.dir_sprint0`;
- the evaluator;
- the SRD2-G0 runner or mechanics.

**Shared code** is limited to:
- hashing and JSON helpers;
- the pinned tokenizer partition;
- the frozen historical R0/S1 region primitives (historical v_AC control only);
- the SRD2-G0 auditor's independent mapping;
- library numerics.

**Own recomputation.** Everything belonging to the new families is recomputed with the auditor's own code:
- D3 candidates, D4 and D0 geometry, random, D5 evidence, prototypes, directions and shuffle;
- v_AC directions;
- the gate (float32 masses);
- energies, top-1 decisions, the severe-EOS proxy, coverage, counts, the bootstrap, predicates and the label.

**Opened-path allowlists.** Both the main process and the phone children are checked: audio of the 540 rows, the Whisper
model, the provider model and engineering clips, pinned sources, the run directory, D1 folds and (Job B) the old30
apparatus. Forbidden: parquet, transcript, CTC/MMS, roles, evaluation files, phonemizer and espeak.

## 4. Claims and locked roles

**Authorized.** Only the 240 D-dev-select utterances produce new scientific outcomes. The FULL300 bank is used unlabeled
for D5 construction only.

**Not touched.** D-dev-confirm, D-test, router-calib, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND and every transfer corpus.

**Not claimed:**
- fresh validation;
- speaker- or dialogue-independent generalization;
- a full-ASR improvement;
- deployable safety.

**Exposure after evaluation.** The 240 utterances become exposed development data. They are recorded as a new
append-only entry in `docs/current/DATA_EXPOSURE_ADDENDA.json`; `DATA_EXPOSURE.md` stays byte-pinned.

**Interpretation limits:**
- Gate-zero and no-edit correct states do not establish safety; actively edited denominators are reported separately.
- A single-token EOS proposal is a risk proxy, not a measured truncation.
- Large logit shifts, script-mass changes or positive margins alone are never success.
