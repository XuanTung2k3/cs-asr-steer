# DIR-SPRINT0 implementation design

This document implements `DIR_SPRINT0_SPEC.md` and `configs/inference_cf/dir_sprint0.json`. Every reused historical
interface is used unchanged. New code is isolated in the files listed below.

## 1. Reused, unchanged interfaces

| Component | Interface | Constraint |
|---|---|---|
| Population exclusion | `exposure_registry.exposed_ids()` | Hash-checked ledger + addenda; 700 IDs |
| Audio / features | `models.whisper.load_audio`, `batch_model_inputs`; `lexical_compatibility.waveform_model_inputs`, `hard_mask` | Zero-change identity checked per v_AC utterance |
| Cached queries | `inference_cf_p2r.DiagBranch` (= `cached.Branch` + crop), `core_p1.processed_argmax` | Own cache per branch; explicit lowest-ID argmax |
| R2 gate | `srd2_g0.r2_gate` with `core_r2` primitives, `inference_cf_p0_r2.full_replay` / `native_lid` | SRD2-G0 g_old unchanged, including compatibility bounds |
| Native site | `srd2_g0.NativeSite`, `native_preview`, `chord_guards`, `CachedSolver`, `tensor_sha`, `bf16_bits` | Passive observers; lossless BF16 storage |
| D2 | `directions.ReadoutDirection` → `readout.readout_direction` | Unchanged; old30 apparatus reproduces P2-DIR bitwise |
| D3 gradient mechanics | `readout.ZeroProbe`, `clone_scratch`, `cache_fingerprint`, `tangent_unit` | Only the objective differs (z[c] − z[b]) |
| D0 / D4 tangent | `core_p1.direction`; `readout.tangent_unit` | Float64 geometry |
| D1 | sealed `results/inference_cf/p2dir/folds_run1` vectors | Hash-verified, no refit |
| v_AC | `r0_regions` (grid, classify, vote, track), `s1_evidence.heard_attention` / `select_target`, `src_cf0_pilot.direction` | Frozen R0 and S1 configurations copied verbatim into the config |
| Phone provider | `phone_provider.PhoneProvider`, `interval_weights`, `FeatureTable` | Manifest- and table-digest pinned |
| Pulses | `p2r.solve_scale`, `scaled_direction`; `loc0_sites.pulse_action`, `cross_pulse_hook` | Engine copied verbatim from the SRD2-G0 runner |
| Mapping (evaluator) | `inference_cf_srd2_g0_evaluate.map_utterance` | Unchanged SRD2-G0 evaluator mapping |
| Mapping (auditor) | `inference_cf_srd2_g0_audit.own_map` | Independent implementation |

## 2. New, isolated components

- **`src/csasr/inference_cf/dir_sprint0.py`** (mechanics, reference-free):
  - the runtime projection;
  - the D3 legal set, candidate rule and `token_margin_direction`;
  - D4, D0 and random;
  - D5 `concept_evidence`, `build_prototypes`, `d5_direction` and `d5_shuffle`;
  - the v_AC wrappers;
  - `arm_target`.
- **`experiments/inference_cf_dir_sprint0.py`** (runner): prepare / manifest / capture (+ `phones` children) /
  construct / seal-a / authorize-b / pulses / seal-b, plus a CPU engineering smoke on public engineering clips.
- **`experiments/inference_cf_dir_sprint0_evaluate.py`** (post-seal evaluator; never imported by the runner).
- **`experiments/inference_cf_dir_sprint0_audit.py`** (independent auditor: pre / a / primary / full / concepts-check).
- **`experiments/inference_cf_dir_sprint0_population.py`** (identity selector) and
  **`experiments/inference_cf_dir_sprint0_d5_concepts.py`** (concept table, threshold engineering, provider
  reference).
- **`slurm/inference_cf_dir_sprint0.sbatch`:** PHASE=capture|pulses; 03:00:00; MIG 3g.40gb; 12 CPUs; 96 GB; no
  chaining.
- **Tests:** `tests/test_dir_sprint0_impl.py` and `tests/test_dir_sprint0_freeze.py`.

## 3. Job A algorithm

1. **Guards.** Verify the committed and pushed manifest, the pushed `PASS_TO_DIR_SPRINT0`, and every source, config,
   population, model and provider pin.
2. **Phone children.** Spawn two CPU children (`phones --which prospective|bank`, 4 threads each). Each first runs the
   provider apparatus on the two engineering clips, then the full-audio posteriors of its rows:
   - `phones/UID.npz` and `.json`;
   - `phones_runtime_*.json` with its opened-path log.
3. **Model.** Load Whisper; compute the null LID, verified against the historical pins, and the null-audio encoder output
   (`zeros(480000)`).
4. **Per prospective utterance:**
   - audio hash check; encode;
   - B0 cached greedy (with heads) and the teacher-replay bitwise check;
   - R2 full replay with heads and site recorder, then per-query `r2_gate` and compatibility;
   - E, TL and NULL branch replays with prefix-identity flags;
   - R0 window-grid LID (shared LID cache) and track;
   - per structural query: the D3 decision and the S1 v_AC target;
   - one cold masked replay per distinct target mask;
   - causal probes;
   - one lossless NPZ and one JSON row.
5. **Per bank utterance:** B0 cached greedy, the full replay for R2 windows, and the B0 sites (NPZ).
6. **Close-out.** Wait for the children and write the runtime: weights digest start/end, gradient counters, VRAM and
   opened paths.
7. **Construct (CPU, deterministic, in the allocation).**
   - D5 evidence for the bank and prospective queries; prototypes and diagnostics.
   - Per prospective utterance: a `construct/UID.npz` with D0, D1, D4, D5, D5SH, VAC and RND per structural query (NaN
     rows for invalid entries), and a JSON of per-query statuses.
   - `shuffle.json` and `construct_summary.json`.
   - If construct raises, the capture is preserved and the documented `construct` CPU mode reruns it unchanged. The
     failed directory is kept.

## 4. Job B algorithm

1. **Old30 historical apparatus** (P2-DIR spot set; outside every denominator). This is the SRD2-G0 bitwise
   reproduction: clean logits and site, D2 direction, J, pulse logits and FFN state, zero dose and restore.
2. **Per utterance:**
   - encoder hash equals Job A;
   - per query:
     - at structural queries, from the pre-step cache: D2 (always) and D3 (only for `candidate`);
     - the clean step, bitwise against the Job A logits and site;
     - the D2 and D3 scratch identity;
     - the 15 arms in `arm_order`: each its own solve and preview, crop back to the pristine pre-step cache, then a step;
     - a restore check;
     - advance with the B0 token.
   - Lossless storage: D2 and D3 directions and gradients; native q/u/r; executed FFN inputs; executed full-vocabulary
     BF16 logits.
3. **Directions** come from the sealed construct arrays (hash-verified), except D2 and D3, which are computed here.

## 5. Schemas and sealing

`job_A_seal.json` lists every capture, bank, phone and construct file:
- its SHA-256 and the content-addressed archive copies under `/mnt/data/tungnx/cs-asr-steer/archives/dir_sprint0/run1`;
- the Job A metrics and predicates;
- the reference-free family construction status;
- the critical flags.

`authorization_B.json` contains only the eight frozen fields. `pulse_seal.json` covers every pulse row and array. Runner
inputs never include dialogue, roster, reference or label fields.

## 6. Adversarial design review (performed before freeze)

| # | Risk raised | Resolution in the freeze |
|---|---|---|
| 1 | D3's unconstrained argmax promotes tokens with near-zero null probability | Plausibility window log 1e-3 relative to the B0 maximum (actuator reach), plus a null floor after the log-softmax |
| 2 | A fixed Top-K (S1) excludes most English-confusion targets | Entropy-adaptive window; the rationale is mechanism-based and outcome-free |
| 3 | D3 could edit at every correct Mandarin state, since c_AP ≠ b_t whenever the contrast prefers another token | Accepted. The direct comparator D3CD and the safety predicates measure it; no gating change |
| 4 | D4's translate task targets English, so TR−TL may point away from English corrections | Disclosed; the sign stays fixed |
| 5 | D5 CTC posteriors are blank-dominated, and a diffuse segmental floor (~5%) would leak into q | Emitting-frame rule s_j ≥ 0.5 (bimodal engineering evidence), excluding the floor, plus a minimum emitting weight of 2 |
| 6 | Voicing and aspiration encode language through espeak conventions | Excluded from the primary concepts; descriptive only |
| 7 | panphon codes glottals as dorsal-like ([+back]) | DOR requires +hi; glottals and uvulars fall outside the primary classes |
| 8 | Concept prototypes may encode Mandarin-ness (nasal codas) rather than phonology | Diagnostics: cosine with the bank baseline-script axis, script-class means and dialogue η². No residualization (that would be a post-hoc choice) |
| 9 | The shuffle may be degenerate when adjacent tokens share a window | Derangement (no fixed points for n ≥ 2); identical-z and cos > 0.9 cases reported |
| 10 | Crop phone inference differs from the full audio (max 11.8 nats) | The provider always runs on the full audio; window selection by sample overlap |
| 11 | The provider's CPU numerics may differ between hosts | In-job repeat bitwise; reference within 1e-3 and argmax 1.0; the auditor's spot recomputation uses the same tolerance |
| 12 | A gated variant can qualify a family, which raises multiplicity | Predeclared; disclosed; matched gated comparators (D2G, RNDG, D5SHG) |
| 13 | Gated arms edit few Mandarin states, so safety is not estimable | Estimability gate (≥ 40 edited ZH-correct in ≥ 8 dialogues). Without it, a powered variant is INSUFFICIENT_COVERAGE, never "safe" |
| 14 | No-edit cells credited as safety | Coverage gates on structural and EN-confusion denominators; blocked families are never scored |
| 15 | The bank shares the 20 dialogues with the prospective utterances | Allowed: unlabeled. Disclosed as a speaker/dialogue overlap |
| 16 | Job A wall time, given CPU provider throughput | Two parallel children; conservative 6,000 s; hard 3 h |
| 17 | A construct bug after Job A would waste the job | Construct is deterministic CPU code over sealed arrays and can be rerun unchanged without a third GPU job |

## 7. Tests

`tests/test_dir_sprint0_impl.py`:
- the runtime allowlist;
- the UTF-8 boundary rule;
- the D3 rule (window, floor, ties, EOS, suppression, abstention);
- primary-versus-auditor D3 agreement;
- the token-margin gradient (scratch identity, cache isolation, no parameter gradients, finite difference) on a tiny
  random Whisper;
- D4 sign and tangent; D0; random;
- D5 evidence (silence, spikes, excluded mass, frames, nonfinite);
- window weights; prototypes, including recovery and auditor agreement; D5 direction; shuffle derangement;
- frozen constants against the config;
- arm targets;
- family and terminal precedence;
- the static firewall.

`tests/test_dir_sprint0_freeze.py`: the freeze contract (hashes, population, exposure, legal set, concept digest,
precedence, pins).

**CPU engineering smoke** (`smoke` mode). It runs on public engineering clips only, in a scratch root, and exercises
capture, phone children, construct and pulses end to end.
