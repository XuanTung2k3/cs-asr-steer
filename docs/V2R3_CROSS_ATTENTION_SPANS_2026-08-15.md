# Cross-attention pseudo-label spans on v2r3 — a negative result

**Date:** 2026-08-15
**Job:** Slurm 40711, `COMPLETED`, exit 0, wall clock **142 s** (after 40703 failed; §3)
**Destination:** `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/cross_attention/generation_001`

**Outcome: the preregistered configuration does not produce usable language
spans.** A per-frame argmax over raw final-layer cross-attention assigns **93.2%
of valid frames to the special-token class**, yields English spans with a median
duration of 140 ms, and disagrees with every other path by a median of
**2.8–3.2 seconds**. This is reported as measured. No parameter was tuned to
improve it, no path was selected, and no gate was evaluated.

---

## 1. Purpose and scientific boundary

Following Liu et al., IEEE TASLP 2025 (arXiv 2403.05887): CTC token spans exclude
blank frames and a blank frame carries no language attribute, so a CTC switch
boundary is defined only up to the unowned blank run. That paper derives language
spans from decoder cross-attention instead. This session built that third path
and measured it against the four CTC conventions and `whisper_dtw`.

The model was frozen — one teacher-forced forward pass under
`torch.inference_mode`, no parameter update. **No DTW and no monotonic path**:
each encoder frame takes the language of its argmax token. Roles were
`D-construct` and `D-dev-select` only; `D-dev-confirm`, `D-test`, and every gate
generation were untouched. This run **selects no aligner path, sets no floor,
evaluates no gate, and declares no readiness.**

---

## 2. Frozen configuration

Set in the module source before the first run and recorded in every manifest.
None was chosen by looking at results, and none was changed after seeing them.

| Parameter | Value |
|---|---|
| Decoder layer | `-1` (final of 32) |
| Head set | `all_heads_mean` — all 20 heads of that layer, averaged |
| Smoothing | mode filter, **9 frames = 180 ms** |
| Minimum span | **5 frames = 100 ms** (deliberately below the 200 ms tier so it cannot pre-empt a floor) |
| Special-token handling | own class, never assigned a language; a frame whose argmax is special cannot join a span. Punctuation/whitespace form a fourth `neutral` class with the same protection |
| Prefix | `<\|startoftranscript\|><\|zh\|><\|transcribe\|><\|notimestamps\|>` |
| Frame assignment | argmax over tokens of head-averaged cross-attention |
| Teacher forced / parameters updated / uses DTW | yes / **no** / **no** |

The Whisper alignment-head subset was deliberately *not* used: it is tuned for
timing under DTW, the mechanism this path exists to avoid.

---

## 3. Identity handling and the failed first attempt

**Inputs read — legacy format, byte-verified, identity never checked.** All v2r3
sidecars gate on the combined `source_sha256`, which the tree has moved past;
attempting identity would refuse exactly as job 40373 did. Verified against the
recorded hashes:

| Input | Recorded sha256 | Byte check |
|---|---|---|
| candidate table | `cdc0ff4e…` | ok |
| `role_D-construct.parquet` | `d61c305e…` | ok |
| `role_D-dev-select.parquet` | `c0f3ad12…` | ok |

This follows the precedent in `_parents`, which verifies role manifests by bytes
with `require_identity=False` and demands identity only for the freeze.

**Artifacts published — split format**, carrying both `code_config_sha256` and
`test_sha256`. These are the first artifacts in the repository published under
the split identity.

**Job 40703 failed** (`FAILED`, exit 1, 61 s) before writing anything. The GPU
work had fully succeeded — all 600 utterances processed — and the run then raised
in this session's new module:

```text
conv.switch_transitions() returns tuple[DataFrame, dict]; the whole tuple was
passed as transitions=, so apply_convention indexed a tuple with a string.
TypeError: tuple indices must be integers or slices, not str
```

The destination directory **did not exist at all** — the failure preceded every
write, so there was nothing to clear before retry and no publication had
occurred, meaning the code freeze had not yet opened. The fix was to unpack the
tuple. Before resubmitting, the failed path was exercised end to end on CPU
against the real candidate table for all four conventions plus `whisper_dtw`, and
an injected +50 ms start shift returned a 50.0 ms median against
`blank_excluded` — validating the overlap matching and the disagreement
arithmetic independently of the GPU.

---

## 4. The five real-model checks

### 1. Frame rate — as expected

| Quantity | Value |
|---|---|
| Expected ms per frame | 20.0 |
| Empirical median | **20.0009** |
| Empirical min / max | 19.961 / 39.998 |
| Encoder frames returned | 1500 for every utterance |

Empirical is `duration / valid_frames`, so it equals the expected step up to the
rounding of duration onto whole frames; the 39.998 maximum is a single very short
utterance rounding to one frame. The encoder always returns 1500 frames because
Whisper pads every input to 30 s. **Matches the expected frame rate.**

### 2. Attention mask — the finding that explains much of the rest

| Quantity | Value |
|---|---|
| Encoder attention mask exists | **No** |
| Mean attention mass beyond the utterance's own duration | **0.798** |
| Max over utterances | **0.942** |
| Decoder pad query positions seen | 20,048 |
| Decoder pad attention mass excluded | 20,047.96 (≈1.0 per padded row, i.e. all of it) |

**What was observed:** Whisper applies no encoder attention mask. Every input is
padded to 30 s and all 1500 frames are attended, so on average **80% of
cross-attention mass lands on silence beyond the audio**, reaching 94% for the
shortest utterances. Restricting to `round(duration / 0.02)` frames before any
argmax is therefore load-bearing, not hygiene. On the decoder side the pad mask
behaves correctly: padded query rows were excluded by slicing each sequence to
its true token count, and the mass they carried (essentially one full row of
attention each) never entered the argmax.

### 3. Special-token frames — 93.2%, and it is **not** a mapping bug

| Class | Frames | Fraction |
|---|---:|---:|
| `special` | 436,155 | **93.20%** |
| `ZH` | 14,221 | 3.04% |
| `EN` | 12,741 | 2.72% |
| `neutral` | 4,874 | 1.04% |
| total | 467,991 | |

This is the "implausibly high value" the check exists to catch, so the token
mapping was audited directly on CPU over 50 utterances:

| Class | Share of **tokens** |
|---|---:|
| `ZH` | 36.79% |
| `EN` | 32.46% |
| `special` | 21.24% |
| `neutral` | 9.52% |

Exactly **5 special tokens per utterance** (4 prefix + EOT), the designed number,
with a median of 23 tokens per utterance — so 21% special is arithmetic, not
error. Spot-checked decodings are correct: `<|startoftranscript|>` → special,
`没有` → ZH, `' '` → neutral, `<|endoftext|>` → special.

**Conclusion: the mapping is correct.** 21% of tokens are special, yet they win
the argmax on 93% of frames. That is the attention-sink phenomenon — Whisper's
prefix and EOT tokens absorb the bulk of raw cross-attention mass — not a
token-mapping defect. The same sink behaviour appears on the frame axis in
check 2.

### 4. Spans and coverage

442 embedded-English spans survived (D-construct 223, D-dev-select 219).
Duration distribution, in ms: min 100, Q1 120, **median 140**, Q3 180, P90 258,
max 520.

| Role | Floor | Spans | Utterances | Dialogues | Speakers |
|---|---|---:|---:|---:|---:|
| D-construct | ≥400 ms | **2** | 2 | 2 | 2 |
| | ≥300 ms | 11 | 8 | 5 | 7 |
| | ≥200 ms | 37 | 25 | 10 | 13 |
| D-dev-select | ≥400 ms | **3** | 3 | 2 | 2 |
| | ≥300 ms | 15 | 9 | 8 | 8 |
| | ≥200 ms | 45 | 21 | 12 | 15 |

For comparison, the CTC path yields 409 spans at ≥400 ms in D-dev-select across
20 dialogues. This path yields **3, across 2 dialogues**. The matrix-preceded,
non-final and contiguous-index conditions are defined on reference-unit
sequences; this path derives spans from frames and carries no unit index, so only
the duration floor is applied and the remainder are not applicable.

### 5. EN−ZH signed median difference

| Compared with | EN−ZH (ms) |
|---|---:|
| `existing_ctc/blank_excluded` | −2996.3 |
| `existing_ctc/blank_to_preceding` | −2519.9 |
| `existing_ctc/blank_to_following` | −2481.9 |
| `existing_ctc/blank_midpoint` | −2612.2 |
| `whisper_dtw` | −2480.0 |

Against CTC's **−71.7 ms** under `blank_to_preceding` (Session 8), this path's
asymmetry is roughly 35× larger.

---

## 5. Three-way comparison

Greedy one-to-one matching by temporal overlap within utterance and language —
preregistered, and chosen because positional matching breaks as soon as one path
finds a run the other misses.

| Path | n | Median (ms) | P90 (ms) | ≤100 ms | ≤200 ms |
|---|---:|---:|---:|---:|---:|
| `existing_ctc/blank_excluded` | 390 | 2768.5 | 8957.7 | 0.00% | 1.28% |
| `existing_ctc/blank_to_preceding` | 439 | 2833.9 | 8855.1 | 0.23% | 1.14% |
| `existing_ctc/blank_midpoint` | 414 | 2983.4 | 8937.9 | 0.00% | 0.24% |
| `existing_ctc/blank_to_following` | 406 | 3207.3 | 9081.5 | 0.00% | 0.25% |
| `whisper_dtw` | 449 | 3160.0 | 9044.0 | 0.67% | 1.78% |

Every figure is `cross_aligner_disagreement` between two independent automatic
paths on natural speech — never error against truth.

For scale: CTC and Whisper disagree with **each other** at a 340–694 ms median
depending on convention. This path disagrees with **both** at ~3 s, and ≤100 ms
agreement never exceeds 0.67%. The three-way agreement Day 3's selection rule
requires does not exist in this configuration.

---

## 6. What this does and does not establish

**Establishes:** this specific preregistered configuration — final layer, all
heads averaged, raw per-frame argmax, no sink handling — does not recover
language spans from Whisper-large-v3 on this corpus. The mechanism is visible in
the checks: attention sinks dominate both axes, the token axis (93% of frames
choose one of 5 special tokens) and the frame axis (80% of mass on padding).

**Does not establish:** that cross-attention pseudo-labels cannot work. Candidate
differences from the cited method, none of which were tried, and none of which
should be tried by adjusting this run's frozen parameters after seeing its
results:

- renormalising attention over **text tokens only**, excluding special tokens
  before the argmax rather than after;
- excluding padded encoder frames before normalisation, not just before argmax;
- a different layer, or a head subset selected on an independent criterion;
- attention-sink correction of the kind now standard for this failure mode.

Each is a different method, and each would need its own preregistration and its
own run. Tuning the current parameters until the numbers improve is exactly the
move the repository forbids, and was not made.

---

## 7. Artifacts and verification

| sha256 | Artifact |
|---|---|
| `9acb1df7f35323578d3f7ab43d708e7c…` | `cross_attention_spans.parquet` |
| `4513c1cd0fd1ded97ac6e6994cd1fae5…` | `cross_attention_utterances.parquet` |
| `5103b26b09c39ee3b76da402b3b57ee0…` | `cross_attention_report.json` |

All three published in **split** identity format —
`code_config_sha256 = 407952ce…`, `test_sha256 = 21983432…` — with
`candidates_all.parquet` as recorded parent and `taint_reasons: []`.

### Code freeze

| Checkpoint | `code_config_sha256` |
|---|---|
| Frozen baseline, before publication | `407952ce9f82720eb104b43e9a90c66575dbe65d6e362bea08ff982a6b3c78b6` |
| Printed by the launcher inside the job | identical |
| After completion | identical |

The freeze covered `src/` and `configs/` only, per the narrowed identity binding.
`tests/` was not frozen and did not need to be: the artifacts are split-format.
Full CPU suite before the freeze: **667 passed, 0 failures**.

### Immutability

| Root | Newest mtime |
|---|---|
| `artifacts_lss` | 2026-08-12 02:41:33 |
| `artifacts_v2` | 2026-07-28 10:46:20 |
| `artifacts_dialogue_v2` | 2026-08-14 17:00:10 |
| `artifacts_dialogue_v2r2` | 2026-08-14 19:41:13 |

Only `artifacts_dialogue_v2r3/cross_attention/` was added. Exposure ledger mtime
`1786502490`, sha256 `46849f5caed62201…` — unchanged; no gate generation
allocated, exposed, or consumed.

---

## 8. Open items

1. **The third aligner path does not exist yet.** Day 6's invariance battery
   assumes three paths; there are two. Whether to attempt a corrected
   cross-attention method, and which correction, is a scientific decision.
2. `whisper_dtw` remains the weaker of the two working paths — 17.4% invalid
   units, all ZH, costing the conservative subset nothing (Session 8).
3. The 442 spans are retained as evidence of the failure mode, not as usable
   labels. Nothing downstream should consume them.

**No path was selected. No gate was evaluated. Production Gate A is unchanged.**
