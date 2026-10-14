# CS-ASR-ReFT P0/P1 — result

| Stage | Verdict |
|---|---|
| P0 (Whisper Auto beam-5, ACL-0 dev150) | **`P0_PASS_RELEVANT_OPPORTUNITIES`** |
| P1 (frozen rank-4 gradient subspace, **decoder layer 31**, AMENDMENT L31) | **`P1_SUBSPACE_WEAK_OR_NARROW`**: every frozen gate passes except preservation |
| P1 at layer 24 (original specification) | historical, **no outcome**: capture and basis built, held-out evaluation never completed |
| P2 | not started |

**Scope of the evidence.**
- Exploratory. P0 uses an exposed development panel (A3-ACL0-DEV150).
- P1 uses a dialogue-disjoint audit partition of training data (6 dialogues).
- No independent test was opened.
- P1 measures retention and oracle local controllability. It does not measure free-decoding ASR.

All private artifacts are under `/mnt/data/tungnx/cs-asr-reft/`. Dialogue IDs are replaced here by rank labels.

## 1. Setup and provenance

| Item | Value |
|---|---|
| Model | Whisper-large-v3 at `/mnt/data/tungnx/whisper-large-v3`; `model.safetensors` sha256 `a8e94b85…`; bf16 parameter hash `0b2acef4…` (equal to ACL-0); BF16 eager attention on all 96 modules; frozen (0 trainable parameters; hashes equal before and after every job) |
| Software | Python 3.11.9, torch 2.10.0+cu128, transformers 4.57.6; H100 80GB MIG 3g.40gb |
| Decode (F1) | `generate(language=None, task="transcribe", num_beams=5, do_sample=False, temperature=0, max_new_tokens=200)`, batch 1; language = argmax over language tokens at `<|sot|>` |
| P0 data | ACL-0 dev150 (`panels.json` sha256 `6741eb36…`): 150 utterances, 20 D-dev-select dialogues, exposed |
| P1 data | 500 new utterances from `loc-train ∪ util-train` (27 dialogues; disjoint from the ACL-0 train panel, D-dev-select and the P0 panel): construction 391 / 21 dialogues, audit 109 / 6 dialogues; seed 42 |
| Never opened | D-test, D-dev-confirm, router-calib |
| Freezes | `p0.json` + `p0_r1.json`; `p1.json` + `p1_r1..r3.json` (layer 24, historical); `p1_l31.json` (active) |

### Amendments, all timestamped in `P0P1_PROTOCOL.md` §12

| # | What | Affects outcomes? |
|---|---|---|
| r1 | P0 audio-identity check used the wrong hash convention. Attempt 1 was BLOCKED on that check alone; the record is kept. | No: integrity instrument only, no rerun |
| r2 | P1 data-audit `KeyError` | No: before any P1 output |
| r3 | float32 vector algebra moved to CPU (`cublasSdot` SIGFPE) | No: before any P1 outcome |
| r4 | the env `lib/` held CUDA 12.1 cuBLASLt/cudart, which shadowed torch's 12.8 libraries; pip CUDA libs now come first, and loaded libraries are asserted | No: before any P1 outcome |
| L31 | owner amendment: single site at decoder layer 31 | Prospective for every P1 feasibility outcome (see §3.1) |

## 2. P0 — Whisper Auto baseline

**Integrity: 12/12 checks pass.**
- forced-`<|zh|>` with the same engine reproduces sealed ACL-0 B0 beam-5 on 3/3 utterances;
- `language=None` equals the explicit detected-language prompt on 10/10;
- 10/10 repeat decodes are identical;
- 150/150 detected languages equal an independent argmax;
- all 150 decodes terminated, with 0 cap hits and 0 loops;
- audio 150/150 in the manifest convention, files unmodified since before the decode.

The run took 1.04 s per utterance and peaked at 4.5 GB.

**Strata** come from reference units (fillers excluded):

| Stratum | Utterances | Dialogues | Substantive EN units | Mandarin units |
|---|---:|---:|---:|---:|
| Genuine CS | 97 | 19 | 758 | 4,462 |
| English-only | 18 | 17 | 399 | 0 |
| Mandarin-only | 22 | 15 | 0 | 915 |
| Neither (fillers only) | 13 | 9 | 0 | 0 |

### Corpus metrics, beam-5

AUTO is new. B0, H1 and A1 are sealed ACL-0 outputs. H1 and A1 were trained for the forced-`<|zh|>` prompt, so their
comparison with AUTO is descriptive, not matched.

| | AUTO | B0 forced-zh | H1 | A1 |
|---|---:|---:|---:|---:|
| MER, all 150 | **0.118** | 0.181 | 0.116 | 0.110 |
| English WER / PIER | 0.205 / 0.191 | 0.503 / 0.495 | 0.345 / 0.202 | 0.268 / 0.202 |
| Mandarin CER | 0.095 | 0.108 | 0.062 | 0.071 |
| Substantive EN errors | 215 | 565 | 229 | 229 |
| Non-filler Mandarin errors | 417 | 373 | 255 | 252 |
| CS MER | 0.121 | 0.128 | **0.083** | 0.104 |
| CS English PIER | 0.224 | 0.319 | 0.173 | 0.173 |
| CS Mandarin CER | 0.096 | 0.089 | 0.060 | 0.071 |
| English-only MER | 0.135 | 1.103 | 0.643 (one loop) | 0.259 |
| Mandarin-only CER | 0.079 | 0.074 | 0.069 | 0.068 |

Dialogue-block bootstrap of ΔMER (95% CI):
- AUTO − B0 = −0.063 [−0.112, −0.022];
- AUTO − H1 = +0.002 [−0.056, +0.046];
- AUTO − A1 = +0.008 [−0.039, +0.043].

**Language ID.**
- Overall: zh 104, en 41, ko 3, ru 1, fi 1.
- CS: zh 83, en 13, ko 1.

**Unit-level contingency** over 1,157 substantive English units:
- **Units that H1 corrected relative to forced-zh B0 and that AUTO already gets right: 299 / 368.** Of these, 229 are in
  English-only utterances and 70 in CS utterances.
- Units H1 corrected that AUTO still misses: 69 (62 in CS).
- AUTO wrong while H1 is right: 75 (67 in CS). AUTO right while H1 is wrong: 89 (29 in CS).
- Against B0, AUTO introduces 55 new Mandarin errors and repairs 11.
- Against H1, AUTO has 187 more new Mandarin errors.
- Against B0, AUTO is better on 19 utterances and worse on 15. Against H1 it is better on 26 and worse on 59; in the CS
  stratum, 15 and 42.

### P0 opportunities (frozen gate)

Target errors are substantive EN unit errors in genuine-CS utterances, non-derailed, of kind translation-like, lexical
substitution, omission or other.

- **122 target errors in 51 utterances and 18 dialogues** (thresholds: ≥ 20 errors, ≥ 5 dialogues).
- The top utterance holds 10.7% (limit 40%), the top 3 hold 23.8%, and the top dialogue holds 19.7%.
- By kind: omission 65, translation-like 28, lexical substitution 27, other 2.
- Excluded: segmentation artifacts 26 and boundary errors 17.
- Strict substitutions-only count: 55 errors in 17 dialogues.
- CS Mandarin errors under AUTO: 359.
- English-only target errors, for reference: 30.

| Dialogue | Target | Translation-like | Substitution | Omission | Other | Strict |
|---|---:|---:|---:|---:|---:|---:|
| D01 | 24 | 1 | 5 | 18 | 0 | 6 |
| D02 | 15 | 3 | 0 | 12 | 0 | 3 |
| D03 | 11 | 1 | 1 | 8 | 1 | 2 |
| D04 | 11 | 7 | 1 | 3 | 0 | 8 |
| D05 | 10 | 6 | 3 | 1 | 0 | 9 |
| D06 | 9 | 4 | 0 | 5 | 0 | 4 |
| D07 | 7 | 0 | 6 | 1 | 0 | 6 |
| D08 | 7 | 1 | 3 | 3 | 0 | 4 |
| D09 | 5 | 1 | 1 | 3 | 0 | 2 |
| D10–D13 | 4, 4, 4, 3 | 2 | 4 | 9 | 0 | 6 |
| D14–D18 | 2, 2, 2, 1, 1 | 2 | 3 | 2 | 1 | 5 |
| D19 | 0 | 0 | 0 | 0 | 0 | 0 |

**Reading.**
1. Auto removes most of the forced-zh translation problem: 229/236 English-only H1 corrections are already right under
   AUTO.
2. Genuine-CS lexical errors remain, spread over 18 dialogues. Omissions dominate, and two dialogues carry 30 of the 65
   omissions.
3. The ACL-0 editors still beat AUTO on CS utterances (MER 0.083 vs 0.121) and on Mandarin, so supervised adaptation
   has headroom beyond the decoding fix. That comparison is not matched.

## 3. P1 — frozen subspace at decoder layer 31

### 3.1 Amendment state

AMENDMENT L31 arrived when the layer-24 run had reached this point:
- smoke (59134), capture (59135) and basis (59136) complete;
- held-out leverage (59137) had crashed serializing a masked `-inf` margin, after 13:46 of GPU time and before writing
  anything;
- analysis (59138) never ran.

**Seen for layer 24:** only coverage and the in-sample spectrum (top-4 energy 18%). **No layer-24 feasibility outcome
exists.** The layer-24 evaluation was not completed, so no 24-vs-31 comparison exists. Layer-24 artifacts remain
untouched in `p1/` (hash record `p1/L24_HISTORICAL_STATUS.json`).

**What layer 31 reuses after hash checks:** the panel, the targets and the site-independent Auto decode (sha256
`488d3783…`). It shares no gradients or bases with layer 24. The layer-31 row population is identical to layer 24's:
1,758/1,758 rows match on utterance, source, position, tokens and partition.

### 3.2 Site and engineering acceptance (layer 31, smoke 59156: all pass)

**Site.** `model.model.decoder.layers[31]`, post-cross-attention residual, pre-FFN:

`h = q + encoder_attn(encoder_attn_layer_norm(q), H_enc)`

**Verified on the real model** (the max error figures are taken over every checked row):
- `final_layer_norm` input = q + s, bitwise;
- site is the last decoder layer;
- block output = h + FFN(h), within 1e-5;
- block output ≠ h;
- the block output, not h, feeds `decoder.layer_norm`, bitwise;
- δ = 0 logits are bitwise equal to the unhooked logits;
- a perturbation at t leaves logits before t bitwise unchanged and changes position t;
- greedy cached decode vs teacher-forced argmax: 1/136 mismatches (bf16 near-tie; limit 2%); a 1-shifted alignment
  mismatches 132/136;
- gradient shape 1280, finite, nonzero;
- autograd vs central finite difference along q̂: max relative error 7.4e-5;
- along a random direction: max error 1.1e-4·‖q‖;
- +q̂ raises the margin and −q̂ lowers it;
- repeated gradients are bitwise equal, with no parameter `.grad`;
- weights unchanged;
- CUDA libraries consistent (pip cuBLAS/cuBLASLt 12.8).

**CPU tests.** 18/18 pass (tiny real Whisper), including the last-layer placement test.

### 3.3 Gradient sample and coverage

Rows are the same for both layers.

| Partition | A rows (free-decoding first divergence) | B rows (teacher-forced gold prefix) | P rows (correct controls) |
|---|---|---|---|
| Construction | **175** (CS 155, ZH_ONLY 19, EN_ONLY 1; 21 dialogues) | 586 | 604 (EN 66, ZH 538) |
| Audit | **45** (CS 36, ZH_ONLY 9; 6 dialogues) | 149 | 199 (EN 28, ZH 171) |

**Audit A categories:**

| Category | Rows |
|---|---:|
| zh lexical substitution | 16 |
| zh omission | 9 |
| translation-like | 5 |
| en lexical substitution | 4 |
| insertion | 3 |
| zh→latin | 3 |
| en omission | 2 |
| other | 3 |

- English decisions in audit A: 11, all from CS utterances. **There are no EN_ONLY audit A rows.**
- First-divergence classes over 500 utterances:
  - lexical token 220;
  - filler first 107;
  - format first 94;
  - format only 31;
  - identical 40;
  - EOS 3;
  - segmentation 5.
- Layer-31 median ‖h‖ is 18.3 and median ‖q‖ is 3.4–3.7.

### 3.4 Frozen basis V₃₁

- **Rows and weights:** basis policy **A-only** (175 construction rows over 21 dialogues ≥ 100/10). Weighted energy
  shares are CS 0.886, ZH_ONLY 0.109, EN_ONLY 0.006; no cap was active.
- **Spectrum:**
  - σ₁…σ₈ = 3.02, 2.48, 2.28, 2.23, 2.12, 2.06, 2.00, 1.90;
  - **top-4 energy 14.5%**;
  - effective rank 108 by entropy, 74 by participation ratio;
  - the correction gradients are high-dimensional and V captures a modest share.
- **Checks:**
  - VᵀV = I to 1.3e-15;
  - finite;
  - shape [1280, 4];
  - `eigh` reconstruction principal cosines 1 − 3e-16.
- **Language direction:** cos(vⱼ, SCRIPT1) = 0.44, −0.25, −0.15, 0.15.
- **Basis sha256:** `f020e93b7dfa5be003cc3cb0755ccaf6ef303ec0b881f67b2b77e89a069292c3`.

### 3.5 Held-out audit results (oracle local diagnostics; ε = 0.10·‖h‖ primary)

| Audit group | n / dialogues | ρ_V (median) | ρ max random | Δm_V | Δm max random | Δm SCRIPT1 | Δm FULL ceiling | correct > competitor: V / R42 / FULL |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A, all | 45 / 6 | **0.087** (0.044) | 0.0034 | **+1.58** | +0.41 | +0.72 | +6.79 | 14 / 4 / 40 of 45 |
| A, CS | 36 / 6 | 0.102 (0.050) | 0.0036 | +1.74 | +0.42 | +0.78 | +6.64 | 12 / 3 / 31 of 36 |
| A, CS English decisions | 11 / 5 | 0.105 (0.082) | 0.0038 | +1.94 | +0.43 | +0.87 | +6.40 | 4 / 2 / 9 of 11 |
| A, CS non-translation | 28 / 6 | 0.106 | 0.0038 | +1.71 | +0.43 | +0.79 | +6.63 | 9 / 1 / 23 of 28 |
| A, ZH_ONLY | 9 / 6 | 0.030 | 0.0037 | +0.94 | +0.42 | +0.47 | +7.40 | 2 / 1 / 9 of 9 |
| A, EN_ONLY | 0 | — | — | — | — | — | — | — |
| A, translation-like | 5 / 3 | 0.115 | 0.0051 | +2.12 | +0.48 | +0.72 | +6.52 | — |
| A, en lexical substitution | 4 / 3 | 0.118 | 0.0035 | +1.89 | +0.41 | +1.14 | +6.15 | — |
| A, zh lexical substitution | 16 / 5 | 0.083 | 0.0039 | +1.44 | +0.42 | +0.56 | +6.62 | — |
| B (gold-prefix), all | 149 / 6 | 0.057 | 0.0034 | +1.34 | +0.43 | +0.68 | +7.31 | 52 / 14 / 137 of 149 |
| B, en lexical substitution | 30 / 6 | 0.032 | 0.0033 | +0.88 | +0.33 | +0.50 | +6.02 | 4 / 1 / 27 of 30 |

**Retention.**
- Mean audit ρ_V = 0.087, about 25× random. The random expectation is 4/1280 = 0.0031; seeds 42–46 give 0.0031–0.0034.
- Bootstrap over 6 dialogues of ρ_V − max-seed ρ_R: **+0.084 [0.056, 0.124]**.
- PCA4 gives 0.020 and SCRIPT1 0.017.
- Concentration: 45% of the V−random retention excess comes from the top 5 of 45 rows (76% within the 11 CS English
  decisions).
- In-sample ρ_V is 0.145, so audit/construction = **0.60**: retention shrinks by 40% out of sample.

**Leverage at ε = 0.10.**
- Δm_V − max-seed Δm_R = **+1.17 [0.92, 1.51]** nats.
- Δm_V > 0 on 45/45 rows.
- All 45 audit A rows start wrong; the median initial margin is −2.06 nats.
- The correct token becomes the masked argmax in 14/45 rows (12/36 CS, 4/11 CS-English).
- V delivers about 23% of the unprojected-gradient ceiling. FULL fixes 40/45 rows at the same norm, but that ceiling is
  per-example oracle information.
- Dose: at ε = 0.05 V flips 7/45 rows and at ε = 0.25 it flips 26/45, with Δm_V growing about linearly
  (0.79 / 1.58 / 3.80).
- Every audit dialogue shows Δm_V > the random mean: 1.29–2.26 against 0.34–0.47.

**Language-preference check.**
- SCRIPT1, the rank-1 Latin-vs-Han readout direction, reaches 45% of V's leverage on CS A rows, below the 0.80 dominance
  limit.
- V keeps its leverage on non-translation CS decisions: 28 rows, +1.71 vs +0.43.
- v₁ correlates with the script direction (cos 0.44). Part of V is therefore a language-preference axis, but it is not the
  whole effect.
- English→English substitutions show positive signal but are scarce: 4 audit A rows. On gold-prefix B rows, V's effect is
  weaker: Δm +0.88, ρ 0.032.

### 3.6 Preservation risk (Test 3, 199 correct audit positions: EN 28, ZH 171)

| ε | V worst-of-8 flip | Random worst-of-8 flip (seed mean / max) | PCA4 | SCRIPT1 | V·c̄ (single mean-correction direction) | Random R·c̄ |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.060 | 0.000 / 0.000 | 0.015 | 0.015 | 0.020 | 0.000 |
| **0.10** | **0.136** (EN 0.143, ZH 0.135) | 0.012 / 0.015 | 0.060 | 0.055 | 0.050 | 0.000 |
| 0.25 | 0.452 | 0.055 / 0.070 | 0.151 | 0.166 | 0.116 | 0.015 |

- Mean worst-direction margin change at ε = 0.10: −1.47 for V, against −0.37 for random (median correct margin +4.78).
- Preservation-gradient retention is ρ_V = 0.074 (EN 0.050, ZH 0.078), against random 0.003. V is as aligned with the
  sensitivities of *correct* decisions as with the errors.

**Gate.** V's worst-of-8 flip fraction (0.136) exceeds max(0.10, 2 × 0.012). **Preservation is concerning.** The same
directions that correct errors also flip about 13% of correctly recognized tokens at the same budget.

### 3.7 Verdict

| Frozen gate | Result |
|---|---|
| Integrity: smoke, weights, basis, m₀ reproduced 393/393, audio, CUDA libraries | pass |
| Coverage: ≥ 30 audit A rows / ≥ 4 dialogues / ≥ 15 CS | pass (45 / 6 / 36) |
| R: retention ≥ 0.031 and CI above max random | pass |
| L: leverage CI above max random, ≥ 60% improved | pass |
| CS signal | pass |
| Not narrow: non-translation CS (n = 28) and SCRIPT1 < 0.8·V | pass |
| Generalization ≥ 0.5 | pass (0.60) |
| **Preservation not concerning** | **fail** (0.136 > 0.10) |

→ **`P1_SUBSPACE_WEAK_OR_NARROW`**: the basis has real held-out local leverage on genuine-CS lexical decisions, beyond
random, PCA and a pure language direction, but its preservation sensitivity is concerning. Under the protocol, no
controller training starts automatically.

**What this does not show:**
- that a controller can recover words;
- that V is sufficient for free decoding;
- that the model can learn where to intervene;
- causal isolation of language interference;
- an advantage over PEFT;
- any compute saving.

**Evidence base:** 6 audit dialogues, 45 primary rows, 11 English decisions. The uncertainty at the dialogue level is
correspondingly wide.

## 4. Compute

| Job(s) | Purpose | GPU wall | Peak memory |
|---|---|---:|---:|
| 59116 | P0 decode | 3:29 | 4.5 GB |
| 59126, 59127, 59128, 59129 | layer-24 engineering failures, gdb diagnosis and retries (r3/r4) | 11:13 | — |
| 59134, 59135, 59137 | layer-24 smoke, capture (decode 8.5 min + gradients 2.2 min) and leverage (crashed) | 25:29 | 7.3 GB |
| 59156, 59157, 59159 | **layer-31** smoke, capture (decode reused) and leverage | **16:55** | 8.7 GB |
| **Total** | | **≈ 57 min** | |

- AMENDMENT L31 added about 17 GPU-minutes.
- CPU steps took seconds each.
- Basis extraction (capture plus SVD) costs about 2.5 GPU-min once a decode exists. That is separate from any future
  controller-optimization cost, which was not incurred.

## 5. Recommended next action (one)

Before any P2 training, run a CPU-only, predeclared **selectivity audit** of the frozen V₃₁ on the existing layer-31
artifacts. The question: can the 4-dimensional state `z = V₃₁ᵀ·Norm(h)` that the proposed controller would read separate
error positions (A/B rows) from correct positions (P rows)?
- Fit on construction rows, score held-out audit AUC.
- Compare against random-subspace `z` and a 1,280-dimensional `Norm(h)` upper bound.

The preservation failure means a 20-parameter controller is only viable if it can tell *when* to act. This audit measures
that directly, at no GPU cost, before committing to P2.
