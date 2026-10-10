# P0 feasibility diagnostic — frozen before GPU output

Source: `INFERENCE_STEERING_IMPLEMENTATION_PLAN.md` in the adjacent planning folder, 2026-09-24. This is a separate, inference-time proposal. The older repository method contract governs reuse of its decoder site, metrics and data roles; it does not redefine this P0 proposal. P0 performs no intervention or P1 decoding.

## Identity and reuse

- Original checkout: `/home/tungnx/cs-asr-steer`, branch `learned-expansion`, committed base `ce5d225f41d66c5d8aadc9df3c734fbc608646db`; extensive unrelated modified/untracked A7 work remains there.
- P0 worktree: `/home/tungnx/cs_asr_steer_inf`, branch `feature/inference-cf-steering`, clean at creation from that base.
- Model: local `/mnt/data/tungnx/whisper-large-v3`, hub ID `openai/whisper-large-v3`; weight and tokenizer SHA-256 in manifest, bf16, eager attention, frozen eval. Decoder L24 (zero-indexed), post-cross-attention residual before FFN, the DG-02 validated site. Reuse `load_whisper`, `batch_model_inputs`, `build_prefix`, and `DecoderPostCrossAttnRecorder`. Canonical metrics remain in `csasr.evaluation.canonical`; this P0 scores no recognition outcome.
- Ordinary ASR: `model.generate(task='transcribe', language='zh', do_sample=False, num_beams=1, temperature=0, max_new_tokens=200, condition_on_prev_tokens=False)`. No hidden prompt or decode change.
- Development role: dialogue-v2r3 `D-dev-select` only. Frozen baseline `results/dg04/results/B0.json`; evaluator-only `poi_D-dev-select.parquet` and role reference select diagnostic positions offline. No confirmation, test, SEAME, ASCEND test, or oracle switch enters inference. Selection hashes and source paths are in the manifest.

## Panel

Seed 240924. English POI rows stratified by frozen baseline `correct` into wrong and correct. Mandarin positions are median Han reference units in the same 300 frozen baseline utterances. Each stratum is ordered by SHA-256 of `seed:kind:utterance_id:unit_index`, then ID/index. Select up to 20 per stratum, one row per utterance globally, priority wrong/correct/Mandarin. Selection counts and missing strata recorded. The offline reference unit character fraction selects a *diagnostic location*; on GPU `floor(fraction * generated_content_length)` is clamped to a real next content token. It is not passed as a true future token or reference word. This coarse mapping limits localization claims. Frozen baseline text must exactly match freshly generated text or the row is skipped.

## Conditions, alignment and states

`c0`: ordinary baseline Mandarin/transcribe/no-timestamps prefix from the canonical `build_prefix`; `cM`: explicit Mandarin/transcribe/no-timestamps from the same function; `cE`: explicit English/transcribe/no-timestamps. The run saves human-readable names and exact token IDs. `c0 == cM` is decided by those IDs, not labels. No invented neutral condition.

CPU-resolved tokenizer IDs before GPU: `c0=[50258,50260,50360,50364]`, `cM=[50258,50260,50360,50364]`, `cE=[50258,50259,50360,50364]`. Thus G1 is expected to degenerate by construction; the GPU states must still confirm the residual.

For each row, run the ordinary greedy baseline once and take its **actual token IDs**. This installed Transformers `generate` returns content tokens only; the forced prefix is implicit and absent from `sequences` (first physical job 54702 exposed the parser error and all state rows were invalid). Strip first EOS, choose the offline-indexed logical token position, and use all preceding generated content tokens as one shared prefix. For each condition, concatenate its prompt with this exact content prefix. All decoder masks are ones. State capture uses full causal replay with `use_cache=False`, so cache positions/history are `0..input_length-1` separately for each condition; no KV cache is exchanged. The query at the last input token predicts the same logical next content token. At the first content token the query is the final prompt token. All branches are greedy lineage 0 with no beam expansion. Record prompt length, full masks, cache positions, logical position, layer/site, tokens, and skip reasons. Encoder attention uses the processor's real padding mask and shared audio representation.

Capture `h0,hM,hE` as raw float32 lists from L24's `r=q+u_source` site. Recompute norms, `delta=hE-hM`, `d=delta/(||delta||+1e-6)`, `rho=2<h0-(hM+hE)/2,d>/(||delta||+1e-6)`, diagnostic `c=max(0,-clip(rho,-1,1))`, `||h0-hM||`, finite and zero status. If token-identical `c0/cM` give residual <=0.02 L2, report `DEGENERATE_BY_CONSTRUCTION`: the original collapse factor is non-identifiable and cannot enter P1. `IDENTIFIABLE` requires distinct serialized conditions and finite distinct states; otherwise `BROKEN`.

## K=1 support and audio control

Under each real-audio condition, take raw vocabulary argmax for one next token: `yE` from `cE`, `yM` from `cM`. Do not use the actual future token in candidate creation or scoring. Score **both fixed candidate IDs under both conditions** with one `log_softmax` next-token distribution per condition. Values are single-token log probabilities (no length normalization needed); EOS/special IDs are retained and flagged. The candidate is one token only, so no truncation beyond K=1 and no postcandidate EOS is scored. Nonfinite values cause explicit skip. If IDs collide, `q_cross=null` and collision is uninformative; `q_own` stays diagnostic. Otherwise `q_own=sE(yE)-sM(yM)` and primary `q_cross=0.5[(sE(yE)-sE(yM))+(sM(yM)-sM(yE))]`. No temperature or threshold tuning.

Sort selected rows by stratum/selection order. Assign every audio the next selected row's audio in a cyclic permutation; the last gets the first. There are no self-pairs. For shuffled scoring keep row identity, prefix, candidates, conditions and equation fixed. Replace only encoder representation with the permuted audio. Save map and all four shuffled scores. No second candidate generation on shuffled audio.

## Predeclared decisions

- G1: `IDENTIFIABLE`, `DEGENERATE_BY_CONSTRUCTION`, or `BROKEN` by exact prompt tokens and state residuals above.
- G2 `PASS` only if every retained row has shared logical position, full-replay mask/cache positions, L24/site and greedy lineage identity; every nonretained row needs a reason. Otherwise `BLOCKED`.
- G3 `SUPPORTED` only with >=90% finite retained coverage, <=50% candidate collision, >=10 noncolliding wrong-English and Mandarin rows, AUROC >=0.60 for wrong English versus Mandarin, and mean paired absolute real/shuffled `q_cross` change >=0.05 across >=10 noncolliding rows. Report correct-English separately regardless. Otherwise `WEAK/BLOCKED`. These fixed operational criteria implement the plan's qualitative coherence/audio-dependence gate; they do not claim calibrated error probabilities.

No layer, K, alpha, temperature or score threshold sweep. P1 may start only after G2 pass and G3 support, and if G1 degenerates only under an explicitly revised evidence-only gate contract. This diagnostic does not itself start P1.
