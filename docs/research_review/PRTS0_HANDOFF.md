# PRTS-0 — feasibility handoff: phonetic-reward-guided test-time steering

Base: `d45bf0ec`. Desk review only; ABTS-0 results unchanged.

## 1. Reusable phone assets

- **Provider.** `wav2vec2-xlsr-53-espeak-cv-ft` @ `2c733782` is installed and hash-pinned, with its own venv
  (`envs/dir_sprint0_d5`). CPU fp32, bitwise deterministic, about 0.16 s per audio second.
- **API.** `phone_provider.PhoneProvider.posteriors()` returns full CTC log-posteriors `[T,392]` (blank = 0, 20 ms
  frames).
- **Cached posteriors.** `dir_sprint0/run1/phones/*.npz` (608 MB) covers all 300 K1 FULL300 utterances plus the 240,
  computed on full audio.
- **G2P.** Mandarin: `pypinyin` 0.55 (acl1). English: **none.** phonemizer/espeak are excluded by design; g2p_en,
  CMUdict and epitran are absent.
- **Alignment.** `torchaudio.functional.forced_align`, and the DTW unit intervals in K1 `selection.json`.

**Is D5 suitable for sequence scoring?** The D5 method is not (concept features; lexically inert). The provider is: its
CTC posteriors score any candidate phone string, with English and Mandarin sharing one inventory and blank. Caveats:
- **Never score crops** (up to 11.8 nats off); slice the cached full-audio posteriors.
- **Language-conditioned labels.** The provider never marks English stops as aspirated, and labels Mandarin b/d/g as
  p/t/k.

## 2. Shortest reference-free reward

**R-LANG (no G2P, possible today).**
- **Score:** Mandarin-convention mass (tone-bearing and cmn-only symbols) versus en-only mass in a window.
- **Evidence:** in D5, the tone-bearing share was 0.28 on ZH-correct queries and 0.013 on EN-confusion queries.
- **Value:** the decoder-independent language evidence K1-F found missing; targets English over Mandarin speech (10/17
  chosen pairs).
- **Limit:** it cannot pick the English word.

**R-LEX (needs G2P).**
- **Score:** `LLR = log P_ctc(EN cand) − log P_ctc(B0 Han block)` on the same window.
- **G2P:** pypinyin mapped to espeak `cmn` symbols; for English, CMUdict plus a roughly 40-entry ARPAbet→espeak `en-us`
  map, with OOV words left unscored.

Rules for both rewards:
- **Tones:** collapse them by logsumexp. Provider tone accuracy is unknown.
- **Polyphones:** logsumexp over pypinyin heteronyms.
- **Alignment:** free-boundary keyword DP (~60 lines numpy) in the DTW interval widened by ±400 ms, removing the 200 ms
  localization loss.
- **Length normalization:** unnecessary for the LLR (shared window). For absolute fit, also report GOP-style
  (log P_ctc − Σ max log p_t) / frames.
- **Weakness:** near-homophone loan pairs, such as cool/酷 *ku4*.

## 3. Reusable hypotheses; dropping R0

- **Hypotheses.** K1 capture on FULL300 gives B0, forced-EN and AUTO tokens, plus alignment heads (478 MB on
  `/mnt/data`; logits are stored as hashes only). There are 5,800 proposals with `en_interval` and B0 blocks; audio
  paths and hashes are in `plan_sealed.json`.
- **Retrospective labels.** K1-F classes from `abts0_k1f_attrition.py`.
- **Without R0.** R-LANG becomes the region gate, and R0 is only a descriptive baseline. That restores the 10
  R0-rejected onsets and makes R-LANG responsible for the 2,458 R0-rejected translations.

## 4. Steering infrastructure

**Reuse as is.**
- `ttls_r1r.EpisodeR1R`/`ttls_hook_r1r`: exact zero start, z at the L16 DG-02 site, masks, AdamW, e* projection.
- `ce_terms`, `preservation_term`, `greedy_decode`, `displacement_r1r`.
- The runner skeleton.
- `Episode.run(loss_fn)` accepts any loss, so a REINFORCE term `−(R−b)·log p(y|z)` fits without kernel changes.

**New code.**
- `phone_g2p` (maps, tested against the `feature_table` symbols).
- `phone_reward` (tone collapse, keyword DP, R-LANG, R-LEX).
- A reward-spliced teacher builder.
- A sampling decode under the hook (RL only).

## 5. Smallest useful E1→E2

| Step | What | Answers / reports | Cost and conditions |
|---|---|---|---|
| **E1-O** | Offline oracle. Score the *reference* English onset against B0's Han rendering on reference-aligned windows for the 39 onsets, plus translation pairs. | Whether the provider separates them at all (AUC). A ceiling only, never a selection rule. | Uses references. |
| **E1-D** | Deployable. Apply R-LANG (and R-LEX once G2P exists) to all 5,800 proposals; label afterwards with K1-F. | CORRECT_GENUINE vs OVER_MANDARIN separation; onset recall /39 (ceiling 25 present, 20 localized); B0-correct overwrites. | CPU and cached data: minutes of compute, 1–2 days of code. Exploratory, because FULL300 is reference-inspected; freeze the threshold here. |
| **E2** | One TTLS-R1R CE arm toward the reward-spliced teacher. Required controls: **direct substitution without steering** (historically steering ≤ substitution), zero-z and random-z. | Steering earns its place only by recovering run continuations or non-candidate words; 14/39 onsets are absent from the forced-EN decode. | About 15 min on MIG 3g.40gb for ≤ 60 utterances. Needs authorization. |

- **RL:** defer (~1–2 GPU-h per 100 utterances; needs utterance-level mixed G2P).
- **Confirmation:** a fresh D-dev-select panel, needing a K1-style capture (about 8 min GPU), posteriors (about 10 min
  CPU) and an exposure addendum.

## Blockers

1. **English G2P.** Pinning CMUdict or g2p_en requires an authorized download. Until then only R-LANG is possible.
2. **Validation.** The pinyin→espeak map is unvalidated, and provider accuracy on code-switched speech is unmeasured.
3. **Candidate ceiling.** The reward only ranks existing candidates, and 14/39 onsets are absent.
4. **No blind panel.** FULL300 and the 240 are both exposed.
5. **GPU determinism** is unverified.
