# BASIS-A — Frozen / No-Training Direction-Construction Ablation

**Status:** FROZEN pre-run. This is an additive development experiment; it does not reopen
DG-00–DG-08. The experiment uses only the existing 300-utterance `D-dev-select` population.
Scientific authority remains `METHOD_CONTRACT.md` and the active v6 proposal.

## 1. Scope and frozen inputs

- Model: `openai/whisper-large-v3`, local revision recorded in the manifest, bf16, `eval()`.
- Dataset: CS-Dialogue, the exact DG-04 300-utterance `D-dev-select` subset, 20 dialogues.
  Subset fingerprint is the SHA-256 of sorted utterance IDs with a trailing newline:
  `sha256:4a4ce18e7a368e60108fe1506ee6a70611d532d0821b376dc1ac8d483e2b4440`.
- Intervention: decoder post-cross-attention residual, pre-FFN, exact tensor
  `decoder_post_cross_attn_residual`, Python layer 24, through
  `DecoderPostCrossAttnInterventionHook`. NormPreserve is on and there is no depth rescale.
- Decoding: free greedy decoding, `language=zh`, `task=transcribe`, `temperature=0`,
  `num_beams=1`, `max_new_tokens=200`, `condition_on_prev_tokens=False`.
- No training, learned/oracle gate, parameter update, beam-5, L16, D-dev-confirm, D-test,
  new dataset, or new model architecture.

## 2. Frozen basis and candidate directions

The inputs are the DG-03 L24 artifacts in `results/dg03/basis/`:

```text
v_raw   = mu_E_correct - mu_M_correct
v_cond  = normalize(E[r(c_E) - r(c_M)])
v_local = normalize(v_raw - <v_raw,v_cond> v_cond)
```

`r` is the exact site state. `R` uses `r_hat=normalize(v_raw)`; `L` uses `v_local`; `C` uses
`v_cond`. The fixed-mixture coefficients are recovered from DG-04 and frozen as
`(a_local,a_cond)=(0.5,0.5)` for both mixtures:

```text
d_RC = normalize(a_local*r_hat + a_cond*v_cond)
d_LC = normalize(a_local*v_local + a_cond*v_cond)
```

Every physical direction is unit-normalized before intervention. The expected direction hashes
(canonical contiguous float64 bytes) are:

| direction | hash |
|---|---|
| raw | `sha256:b637493362f2ba319aef3d24f74f382334508cdc28f79fc7a062cd528cf3b2c8` |
| raw-normalized | `sha256:1bcb2a7a15f665715703b3717e4ef50d23432e4df72d037739d646e689fdc542` |
| local | `sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93` |
| conditioning | `sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991` |
| RC | `sha256:2142a7d4d69a7b54a5596bcca953d1966ce0244f3a6ed5feb1d8b459068be4c9` |
| LC | `sha256:b3aa31d521d79c93fae1c22599b42136d8b6cfbd191846f84cbde487e635a930` |

The frozen L24 nominal scale is `s_24=8.931257754160319`. The dose grid is exactly
`rho ∈ {0.5, 1.0, 2.0}`, with nominal update `rho*s_24`; `rho=0` is B0. The exact hook applies
the same alpha/scale semantics and forced-prefix/eligible-position rules to every direction.

## 3. Reuse and GPU matrix

DG-04 B0, B1 local-only, and B2 local+conditioning are reused only after checking layer, site,
population IDs/fingerprint, decoder, NormPreserve, eligible-position semantics, rho grid, scale,
basis hashes, and canonical `result_v1`. Their paths and byte hashes are copied to
`results/basis_ablation_frozen/reused/` as pointers; the DG-04 files are never overwritten.

The one new GPU process evaluates `R × 3`, `C × 3`, and `RC × 3` after one model/basis/data load.
It also records one deterministic final free-decoding exact-site L24 row per utterance for the
bounded supporting PCA (maximum 10,000 rows). This baseline recording must reproduce the reused
B0 transcripts exactly. There is no projection-distribution analysis unless compatible labels
and states are already available; no new alignment or probe is introduced.

## 4. Metrics and artifacts

Each new condition emits validated canonical `result_v1` using the existing canonical metrics:
MER, PIER, embedded WER (`en_wer`), matrix CER (`zh_cer`), corrections, corruptions, canonical
utility (`corrections-corruptions`), canonical outside harm, embedded/matrix retention, realized
edit count, total/mean energy, and any available displacement. Overall WER is **n/a — no frozen
canonical overall WER**; MER is not replaced by a newly invented WER. Gate coverage remains the
reserved null field. Efficiency quantities are descriptive and include correction/corruption per
energy, utility/energy, and per 1,000 realized edits.

CPU geometry reports pairwise cosines/angles, Gram matrices, singular values, condition numbers,
effective rank, residualization magnitude, projection matrices, principal angles, projection
Frobenius distance, and RC–LC physical angle. PCA is fit to representation rows, never the five
direction vectors. Figures are a simple correction–damage frontier, cosine heatmap, PCA geometry,
and dose-response utility plot; no UMAP/t-SNE.

## 5. Dose-response and interpretation rules

The full three-point rho trajectory is reported for each direction. Descriptive labels are
assigned mechanically: `NON-MONOTONIC` if correction or corruption decreases; otherwise
`DAMAGE GROWS FASTER THAN CORRECTION` when utility falls while corruption rises; otherwise
`SATURATING` when high-dose corrections plateau or utility does not improve; otherwise
`STABLE DOSE RESPONSE`.

For the requested categorical conclusions, matched-rho POI utility is primary, with damage,
outside harm, retention, and efficiency as checks. A direction wins a pairwise comparison only
when it has strictly higher utility at at least two of the three rho values. Residualization is
`RESIDUALIZATION SUPPORTED` if L wins R under that rule, `RAW BETTER` if R wins L, otherwise
`RESIDUALIZATION NEUTRAL`. Conditioning is `CONDITIONING COMPLEMENTARY` if RC beats R or LC
beats L at least twice in aggregate with no matched-rho mixture loss against its corresponding
single direction; it is `CONDITIONING HARMFUL` only if neither mixture wins and C at rho=1 is
strictly worse than both R and L; otherwise `CONDITIONING NEUTRAL`.

The current basis is `CURRENT BASIS SUPPORTED` only if LC beats both R and L at least twice,
`SIMPLER BASIS PREFERRED` if R or L beats LC at least twice, and otherwise
`CURRENT BASIS NOT CLEARLY BETTER`. Geometry/PCA cannot choose a performance winner. These rules
do not consider learned-controller results. The experiment supports only frozen development
comparisons on this exposed selection population; it cannot establish causal superiority,
semantic purity, generalization, confirmation, or test-set performance.

## 6. Batch, preflight, and Slurm

The single launcher uses `#SBATCH --partition=main`, one GPU, 8 CPUs, 96 GB host memory, and a
four-hour wall-time limit. A bounded eight-utterance preflight compares batch 1 and the proposed
batch for Frozen and nonzero Local decoding. Normalized transcripts and canonical metrics must
match exactly; no semantic fast path is accepted. The initial batch-32 preflight (job 51105)
changed Local transcripts, so the correctness-preserving operational value is frozen at batch 1.
This is an execution-path resolution, not scientific tuning; the failed batch-32 preflight is
retained in provenance. Runtime metadata records GPU name, peak allocated/reserved VRAM,
throughput, batch size, runtime, job ID, and terminal state.

## 7. Provenance and data guard

The pre-run manifest is `results/basis_ablation_frozen/run_manifest.json`. It records the frozen
commit/config, model metadata, all direction hashes, scale, coefficients, rho, subset fingerprint,
decoder, metric schema, planned cells, representation protocol, geometry implementation, and
batch policy before scientific decoding. `D-dev-select` is the only evaluation population;
`router-calib` is not read, and no D-dev-confirm or D-test output is produced. The addendum is
additive and does not modify DG-07/DG-08 status or artifacts.
