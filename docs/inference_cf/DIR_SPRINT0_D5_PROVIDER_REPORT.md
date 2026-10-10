# DIR-SPRINT0 — D5 phonetic provider: setup and engineering verification

**Readiness: `DIR_SPRINT0_D5_PROVIDER_READY`.** The independent audit returned `DIR_SPRINT0_D5_PROVIDER_AUDIT: PASS`
(28/28 checks, second attempt; the first attempt is preserved, see section 7). This unblocks the
`DIR_SPRINT0_D5_PROVIDER_BLOCKED` stop of `72727f34`.

Scope of this session:
- provider installation, freezing and engineering verification only;
- no DIR-SPRINT0 design freeze, population selection, steering direction, pulse, Slurm job, CS-Dialogue audio,
  reference, label or lexical outcome;
- engineering audio was synthetic signals and permitted public samples only.

The checks show that the tool behaves plausibly. They do **not** establish phone accuracy on Mandarin–English
code-switched speech.

## 1. Provider

| Item | Value |
|---|---|
| Model | `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, revision `2c733782da5604684829819a5eb744c193fe9398` |
| Training | wav2vec2-large-xlsr-53 fine-tuned on Common Voice for espeak-ng phone labels; paper arXiv:2109.11680 |
| License | Apache-2.0 (verified live on the Hugging Face API) |
| Weights | `pytorch_model.bin`, 1,263,535,127 B, `sha256:04366b6c…f551` (equals the published LFS hash) |
| Output | Frame-level CTC posteriors over 392 symbols; CTC blank `<pad>` = id 0; one frame per 320 samples (20 ms), receptive field 400 samples |
| Feature mapping | panphon 0.22.2 (MIT), 22 segmental features (`hitone` and `hireg` dropped) |
| Local paths | model `/mnt/data/tungnx/cs-asr-steer/providers/wav2vec2-xlsr-53-espeak-cv-ft/<revision>/`; environment `/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5` |

**Choice.** The primary candidate met every requirement, so allosaurus was not evaluated. No downstream steering
result or gold transcription influenced the choice. The MMS grapheme aligner stays rejected, because its 26 Latin
letters are not phones.

**Tokenizer.** The provider's own tokenizer class (`Wav2Vec2PhonemeCTCTokenizer`) is not used. It would need
`phonemizer`, which converts text to phones and is irrelevant here. Ids map to symbols through the pinned
`vocab.json`.

## 2. Environment (no shared state modified)

- **Separate venv.** A user-owned venv was created from the acl1 interpreter with `--system-site-packages`, so it
  reuses acl1's torch 2.10.0+cu128, transformers 4.57.6 and numpy 1.26.4 without changing them.
- **Installed packages.** Only two, hash-pinned with `--require-hashes` from a local wheelhouse; both digests were
  re-verified live against PyPI by the auditor:
  - panphon 0.22.2 (wheel);
  - unicodecsv 0.14.1 (sdist; a panphon dependency missing from acl1).
- **acl1 untouched.** The acl1 `pip freeze` SHA-256 is identical before and after, and acl1 still has no panphon.
  The `pip check` conflicts shown in the venv (opencv/rerun numpy, torch nccl/triton) were already present in acl1
  before installation and are recorded.
- **Other models untouched.** The Whisper-large-v3 files still match the SRD2-G0 pins. Historical model caches were
  not used: downloads went by pinned-revision URL into a new directory.
- **Runtime does not need panphon.** The adapter reads the frozen feature table. panphon is needed only to rebuild or
  audit it.
- **Resources.** Disk: 20 TB free on `/mnt/data`; the provider uses 1.27 GB. Network: Hugging Face, PyPI and GitHub
  reachable; Wikimedia Commons blocked (not needed).

## 3. Verification A–J (`results/inference_cf/dir_sprint0/d5_provider/engineering_checks.json`, all PASS)

| | Requirement | Evidence |
|---|---|---|
| A | Genuine phonetic units | 274 of 388 non-special symbols use IPA-specific characters (local MMS aligner: 0). Distinctions orthography cannot make are present: θ/ð, ʃ/ʒ, tɕ/ts, ʂ/s, ɕ/ʃ |
| B | EN and ZH coverage | All 42 required espeak `en-us` symbols and all 34 required espeak `cmn` symbols are present and mapped (aspirated stops, retroflex/alveolo-palatal/dental affricates and fricatives, apical vowels, ər, ŋ coda). Tone-bearing finals exist for tones 1–5 (18/26/27/12/27 symbols) |
| C | Frame-level probabilities | Float32 log-posteriors `[T, 392]`; rows normalized within 1e-5; T matches the conv formula on every sample. Non-blank mass is the evidence; frames are 73–84% blank-argmax |
| D | Symbol → feature mapping | 355 segmental symbols, each with a mean-over-segments panphon vector. Tie-barred affricates and diphthongs are explicit segment lists |
| E | Explicit handling | 1 blank, 3 special, 1 tone-only, 32 excluded, each excluded symbol with a reason: ambiguous ASCII digraphs, caret/dot modifiers, doubled segments, a bare `ʲ`, unsupported marks, one panphon-unparseable `i.ː`. None is in the required EN/ZH sets |
| F | Distinctions not collapsed | 29/29 contrasts hold (detail below the table) |
| G | 16 kHz mono | Refused with explicit status: 44.1 kHz, 48 kHz, stereo, int16, NaN, under 400 samples. No internal resampling. Float64 input is bitwise equal to float32. 1 s of silence is 100% blank |
| H | Time alignment | Frame j = samples `[320j, 320j+400)`; the R2 encoder frame f = `[320f, 320f+320)`. A frozen 1 s R2 window maps to exactly 50 provider frames. Overlap helpers are given for windows and for per-encoder-frame attention |
| I | Determinism | CPU float32, fixed thread count: repeated calls, a fresh instance and the auditor's independent model load are all bitwise equal |
| J | No hidden supervision | The adapter has no text, label, alignment, timing or language argument and does not import phonemizer (not installed). The table builder reads only `vocab.json` + panphon. The opened-path log shows no CS-Dialogue, role, reference or SRD2 raw file |

Contrasts verified under F:
- aspiration (spread glottis): `p`/`ph`, `t`/`th`, `k`/`kh`, `ts`/`tsh`, `ts.`/`ts.h`, `tɕ`/`tɕh`;
- voicing: seven pairs;
- affricate vs fricative (continuant): `ts`/`s`, `ts.`/`ʂ`, `tɕ`/`ɕ`, `tʃ`/`ʃ`; affricate vs stop (delayed release);
- place: retroflex vs dental, alveolo-palatal vs postalveolar, `s`/`θ`;
- length and nasalization;
- tone separated from segmental features: `ɑ5` = `ɑ2` = `ɑɜ`, while a standalone `ɜ` stays a vowel.

Engineering samples (plausibility only):

| Sample | Output |
|---|---|
| LibriSpeech `sample1` (English) | `ɡ oʊ ɪ ŋ ɐ l ɔ ŋ s l a ʃ i k a n t ɹ i …` ("going along slushy country…") |
| AISHELL-1 example (Mandarin) | `t ɑ5 j i5 ŋ ɡ ai5 j iɛ5 s. i.5 ɕ i5 ŋ x w a5` |

The Mandarin-convention share of the non-blank mass is 0.38 on the Mandarin sample and 0.008 on the English samples.
CPU cost is about 0.16 s per audio second with 4 threads.

## 4. Normalization rules (frozen in the table)

The checkpoint's inventory mixes IPA with espeak-ng notation. Every rule is listed in `feature_table.json`:

| Group | Rules |
|---|---|
| Mandarin mnemonics | `ts.h`→ʈ͡ʂʰ, `ts.`→ʈ͡ʂ, `s.`→ʂ, `i.`→ɻ̩, `i̪`→ɹ̩, `tsh`→t͡sʰ, `tɕh`→t͡ɕʰ, `th`/`kh`/`ph`→aspirated stops, `nɡ`→ŋ, `ər`→ə˞ |
| Kirshenbaum-style leaks | `tS`, `dZ`, `S`, `N`, `t[`, `d[` |
| Notation fixes | ASCII `:`→ː; `ɚ`→ə˞; `ᵻ`→ɪ̈; affricate digraphs → tie-barred forms |
| Tones | A trailing digit 1–5 is a tone. A trailing `ɜ` is tone 3 only when the same stem also occurs with a tone digit, so English `ɜ`/`ɜː` stay vowels |

## 5. D5 interface (`src/csasr/inference_cf/phone_provider.py`)

- `PhoneProvider(manifest, table, device)`: verifies every file hash, then loads the model frozen (eval, float32, no
  gradients).
- `posteriors(waveform, sample_rate)`: returns log-posteriors, frame count, input and output hashes, a status (`ok`,
  `invalid_input`, `too_short` or `nonfinite_output`) and provenance.
- `frames_in_interval` / `interval_weights` / `encoder_frame_weights`: the exact sample-overlap mapping to the frozen
  R2 localizer.
- `FeatureTable` and `feature_evidence(log_probs, weights, table)`:
  - returns q_k, the expected feature value over the segmental (non-blank) mass;
  - reports blank, special, tone-only and excluded mass separately, plus per-symbol valid mass and per-tone mass;
  - falls back deterministically to `no_frames`, `zero_valid_mass` or `nonfinite_posteriors` with `q = None`.

## 6. Limitations the D5 design freeze must address

1. **Language-specific label conventions.** English voiceless stops are never labelled aspirated, though they are
   aspirated in onsets. Mandarin aspiration is labelled, and Mandarin unaspirated b/d/g are labelled p/t/k. So
   spread-glottis and voicing evidence partly reflects the provider's implicit language choice. The engineering
   samples show this: spread-glottis mean −1.00 on English, −0.95 on Mandarin. The mapping is faithful, not
   collapsed, but these features are language-confounded and must be either excluded from D5 concepts or reported as
   such.
2. **Mandarin is covered through espeak `cmn` conventions.** That includes whole tone-bearing finals and an espeak
   `ə` where IPA would use `ɤ`. The model card does not list its fine-tuning languages; the tone tokens imply Mandarin
   data. Accuracy on CS-Dialogue code-switching is unknown.
3. **CTC posteriors are peaky.** 73–84% of frames are blank-argmax, so window evidence rests on a few spikes per
   phone. The minimum valid-mass rule must be frozen in the design. The adapter only reports the mass.
4. **Run on full audio, not crops.** A 1 s crop gave log-probabilities differing by up to 11.8 nats from the
   full-audio slice, with only 86% argmax agreement, because of re-normalization and lost context. The design should
   run the provider once on the full original waveform and select frames by overlap.
5. **Concept features are not yet chosen.** Voicing = `voi` and nasal = `nas` exist directly. Stop, fricative, labial,
   coronal and dorsal must be defined from panphon features in the freeze. Some panphon codings are idiosyncratic,
   for example `ɹ` is +hi +round and `h` is +son.
6. **Determinism is shown on CPU only.** If Job A runs the provider on GPU, it must re-verify determinism there. The
   CPU cost (about 4 min for 240 typical utterances) also makes a CPU pass viable.
7. **Test-time requirements.** Any D5 method needs Whisper plus this 1.26 GB external phone model, the frozen
   feature table and a calibration bank. It is not Whisper-only.

## 7. Audit record

| File | Content |
|---|---|
| `provider_audit_attempt1_FAIL.json` | Preserved. One auditor-only expectation error: its golden list gave the bare tone mark `1` tone `None`, while the frozen table correctly records it as `tone_only` with tone 1 and no features. The auditor expectation was fixed and a check was added that tone-only rows carry a tone and no features. No provider, table, adapter or engineering output changed |
| `provider_audit.json` | **PASS** |

The PASS audit independently covered:
- model, wheel and environment hashes, with live Hugging Face and PyPI checks;
- the phonetic inventory against the MMS grapheme aligner;
- the table digest, and all 355 feature vectors recomputed from panphon;
- tones from its own rule code;
- golden normalizations and panphon contrasts;
- posteriors from its own model load and normalization, bitwise equal to the sealed hashes;
- conv-formula frame geometry;
- firewall properties;
- acl1, Whisper and pinned-ledger invariance;
- the exposure addendum;
- new files only, with no Slurm jobs.

During development one engineering-check expectation was corrected before any audit. Spreading encoder attention onto
the overlapping 400-sample provider frames carries 1.25× mass per interior frame, not 1×. The adapter docstring now
states this. No adapter arithmetic changed.

Tests: `tests/test_dir_sprint0_d5_provider.py` (27 pass in the provider venv; in acl1 26 pass and 1 skips for missing
panphon). Historical contract suites (SRD2-G0 freeze/implementation, S1, SRC-CF0 contract) pass 168/168.

## 8. Exposure tracking

The SRD2-G0 400 utterances are now registered in the new append-only `docs/current/DATA_EXPOSURE_ADDENDA.json`
(entry `A1-SRD2-G0-400`). It records:
- the IDs, with ID hash `sha256:ff2e3054…`;
- the population, run-manifest (`sha256:08940ec9…`) and terminal hashes;
- the commits.

`docs/current/DATA_EXPOSURE.md` is unchanged, so its SRD2-G0 pin still holds.

`csasr.inference_cf.exposure_registry.exposed_ids()` returns 700 IDs: the 300 documented in the SRD2-G0 registry plus
the 400. It hash-checks the ledger, the registry and the addendum, and raises on any edit. The DIR-SPRINT0 sample
selector must use it.

## 9. Recommendation

Resume the full DIR-SPRINT0 design freeze with all three families (D3, D4, D5). The D5 section must freeze:
- the provider (this manifest digest);
- the full-audio frame-selection rule;
- the concept-feature definitions and the language-confound handling of aspiration and voicing;
- the minimum valid-mass rule;
- the calibration bank and its pseudo-label rule;
- the shuffled-feature control.

Nothing here authorizes Job A or Job B.

Artifacts are in `results/inference_cf/dir_sprint0/d5_provider/`:
- `provider_manifest.json`
- `feature_table.json`
- `engineering_checks.json`
- `provider_audit.json`
- `provider_audit_attempt1_FAIL.json`
- `acl1_before_install.json`

Reproduction: `docs/inference_cf/DIR_SPRINT0_D5_PROVIDER_INSTALL.md`.
