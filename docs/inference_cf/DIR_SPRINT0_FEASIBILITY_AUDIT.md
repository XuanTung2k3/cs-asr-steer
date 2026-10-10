# DIR-SPRINT0 — CPU-only feasibility audit (pre-design)

**Result: `DIR_SPRINT0_D5_PROVIDER_BLOCKED`.** Stopped before any design freeze, runner, scientific GPU job or new
data use, as the sprint instruction (section 2.1) requires. D3 and D4 can be built from existing providers. D5 cannot:
no frozen phone-recognition model and no phone-to-articulatory-feature mapping exist on this machine. D5 is **not**
an experimental failure; it was never run. A human decision is needed (section 7).

- Repository: `feature/inference-cf-steering`, local = `origin/cs-asr-steer-inf` = `f2f0282f`, clean, no Slurm jobs.
- Script: `experiments/inference_cf_dir_sprint0_feasibility.py` (CPU only; no model load, forward, audio, role parquet,
  reference, download or install; every opened path is logged).
- Machine-readable record: `results/inference_cf/dir_sprint0/feasibility_audit.json` (hashes of every inspected file).

## 1. D5 phonetic provider — BLOCKED

| Required prerequisite | Found |
|---|---|
| Frozen acoustic phone recognizer (phone inventory, posteriors) | **None.** 21 local model directories classified from their config/vocabulary; none emits phones |
| Phone/IPA → articulatory-feature mapping | **None.** panphon, epitran, ipapy, phonecodes, phonemizer, allosaurus not installed; no espeak/espeak-ng |
| Reproducible preprocessing / audio-window association | Association could reuse the frozen R2 alignment-head window; moot without a provider |
| Unlabeled calibration bank | Available in principle (FULL300 via a frozen pseudo-phone projection); moot without a provider |

The two closest local models were checked and rejected:

- **`/mnt/data/tungnx/models/mms-fa`** (Wav2Vec2ForCTC, MMS-300m forced-alignment checkpoint). Its output inventory is
  31 symbols: four specials, the 26 lowercase Latin letters and an apostrophe. These are romanized-grapheme
  posteriors, not phones. Letter values depend on the language. For example, pinyin `b`/`p` mark an
  unaspirated/aspirated contrast where English `b`/`p` mark voicing, and pinyin `x`, `q`, `c`, `z`, `j` have no English
  letter value. Reading letters as articulatory features would invent a phonetic model, which the sprint forbids, and
  the bias would fall exactly on the EN/ZH contrast under study.
- **`Qwen3-ForcedAligner-0.6B`** needs an input transcript and returns unit timestamps. It has no phone inventory and
  no phone posteriors.

The other local models are ASR, TTS or LLM checkpoints (Whisper-large-v3/turbo, Qwen3-ASR 0.6B/1.7B,
MOSS-Transcribe-Diarize, omniASR-LLM-7B, XTTS-v2, Vevo, Qwen2.5, BART). The HF-cache MMS aligner entry has refs only,
with no weights.

**What would unblock D5.** A human would need to arrange, outside this sprint's automatic actions:

1. A frozen multilingual phone-CTC model with an IPA inventory covering Mandarin and English. The standard choice is
   `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, with an espeak IPA phone inventory. `allosaurus` is a universal-phone
   alternative.
2. An IPA → feature table such as `panphon`, ideally installed into an isolated environment rather than the shared
   `acl1` environment.

Weight and file hashes would be pinned at arrival. The model's quality on CS-Dialogue Mandarin is unknown until it is
run. Any D5 result would then need an external pretrained phonetic model plus calibration data, so it could not be
described as Whisper-only.

## 2. D3 acoustic-prior contrast — implementable

- **Null branch.** The historical null is `np.zeros(30 * sample_rate)` through the frozen preprocessing (R2
  `native_lid`; reused by SRD2-G0). Whisper pads every input to 30 s, so one null encoder output serves all
  utterances. A per-utterance forced-ZH `Branch` is fed the identical B0 content prefix.
- **Gradient.** `readout.py` provides `ZeroProbe`, `clone_scratch`, `tangent_unit` and the scratch-cache guards.
  `objective()` is hard-coded to J = log P_E − log P_M, so D3 needs one new objective (z(c_AP) − z(b_t) on processed
  logits) through the same mechanics. No other change is needed.

**Decisions the design freeze must settle, before any outcome:**

- **Candidate vocabulary.** The plain argmax of S = 2·log p_B − log p_NULL over the full lexical vocabulary is
  unbounded in practice. A token with p_NULL at the ε floor gains about +27.6 nats at ε = 1e-12. c_AP would then often
  be an implausible low-p_B token. Contrastive decoding normally restricts candidates by a plausibility constraint
  fixed in advance (for example p_B ≥ α·max p_B, or a fixed Top-K). Picking one is a pre-freeze choice, not tuning.
  Whether "nonlexical" covers punctuation, whitespace-only and partial-UTF-8 Han byte tokens also needs fixing.
- **Prior evidence (exposed, motivation only).** S1's region-masked version of the same λ = 1 score raised reference
  ranks on 4 paired rows but made 0 top-1 corrections. Union@20 held the reference first token for only 11/60
  EN-confusion queries, at a median rank of about 300–350. MECH-LANG0: D2, also a readout gradient, closes a median of
  only about 27% of the EN-confusion gap at e*. D3 can correct a query only when c_AP is an acceptable reference token,
  and S1 suggests that will be rare.

## 3. D4 transcription-fidelity contrast — implementable

- **Token IDs** verified against the pinned `generation_config.json` and `added_tokens.json`:

  | Prompt | IDs |
  |---|---|
  | transcribe | `[50258,50260,50360,50364]` |
  | translate | `[50258,50260,50359,50364]` (`task_to_id.translate = 50359`) |
  | forced-EN, for D0 | `[50258,50259,50360,50364]` |

  Task IDs only enter as prompt tokens. Their presence in `suppress_tokens` affects generation logits, not prompt
  feeding.
- **Branch.** `Branch(bundle, encoded, prompt, name)` takes any prompt, so a translate branch fed the B0 content
  prefix gives h_TL at the same absolute query index. The tangent projection reuses D2's float64 rule.
- **Adversarial note** (no sign search; the instruction fixes TR − TL).
  - Whisper's translate task targets English. TR − TL therefore partly encodes "do not output English", which at an
    EN-confusion query may point away from the needed English token.
  - MECH-LANG0 already flagged this entanglement with output-language selection. The historical v_prompt result
    (useful sign toward forced-ZH) leaves the expected effect ambiguous.
  - It must be reported as such, not resolved by choosing a sign.

## 4. Historical controls — available

| Control | Status |
|---|---|
| D0 | `core_p1.direction(h_E, h_B)` per query; needs a forced-EN branch |
| D1 | 20 sealed NEW_P2_DIR_CROSSFIT_V1 fold vectors, all status ok, 20/20 vector file hashes verified; fold k is applied to dialogue k (dialogue-held-out), no refit |
| D2 | `readout_direction` |
| R2 g_old | `srd2_g0.r2_gate` + `core_r2` primitives |
| Random | `src_cf0_pilot.random_direction` (PCG64 per UID/t/layer) |
| v_AC | Reference-free, but on new utterances it needs the full R0 native-LID region pass, S1 query association and hard-zero masked encoders. Historically v_AC was valid at only 73/180 queries of an enriched panel, and R0 found at least 6 decoder queries inside predicted-English regions in only 48/300 utterances. All-population coverage will be low and must be reported on both denominators |

## 5. Population — available

The committed SRD2-G0 identity roster has 7,919 D-dev-select utterances in 20 dialogues. It excludes:

- 300 from the documented-exposure registry (FULL300 union);
- the 400 SRD2-G0 utterances.

That leaves **7,219** eligible, with **92 to 895 per dialogue**, against 12 needed. Only identities were counted; no
utterance was selected.

## 6. Compute pre-forecast (not a frozen projection)

This is scaled from measured SRD2-G0 runtimes on the same MIG 3g.40gb type:

- Pulses: mean 0.0134 s.
- Readout autograd: mean 0.035 s.
- Capture: 1,017 s for 400 utterances.

For 240 utterances, about 6,160 structural queries are expected. The forecast assumes the following:

| Item | Assumption |
|---|---|
| Ungated arms | 9, every one pulsing every structural query |
| Gated arms | 4, at the SRD2 nonzero-gate fraction |
| Autograd | 2 calls per query |
| Job A cost | 2× the SRD2 per-utterance capture cost |

| Job | Forecast | Limit |
|---|---|---|
| A | about 1,220 s | 10,800 s |
| B | about 1,580 s (about 60,000 pulses) | 10,800 s |

Both are far below the 3 h limit. A design freeze would replace this with a frozen formula, and Job A would recost it.

## 7. Open items for the human decision

- **D5 provider.** Either arrange one (section 1) and then freeze the full D3/D4/D5 sprint, or authorize a separate
  D3/D4-only study. A D3/D4-only study drops D5, D5·g_old and the D5 shuffled-feature control. All other arms and
  rules stay as specified.
- **Data-exposure ledger.** The next design freeze is where the SRD2-G0 400-utterance exposure must be carried into
  `docs/current/DATA_EXPOSURE.md`. However, `tests/test_srd2_g0_freeze.py` pins that file's exact SHA-256 through
  `configs/inference_cf/srd2_g0.json:source_sha256`, so any additive update fails that frozen historical test. This
  needs an explicit decision: authorize the additive update and accept the superseded pin, or keep the ledger in a
  new file.

No design document, config, population file, runner, evaluator, auditor, Slurm script or test for DIR-SPRINT0 was
created. No existing file was modified. No scientific outcome exists.
