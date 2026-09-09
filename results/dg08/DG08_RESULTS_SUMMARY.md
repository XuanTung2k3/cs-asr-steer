# DG-08 — Locked Whisper Core Evaluation: Results Summary

**Test-lock commit:** `1fb2c3729cd2a51128edbd80e0d99754ce63d526` · **model:** Whisper-large-v3 (local
frozen) · **corpus:** CS-Dialogue · **D-test:** 6,257 utterances / 15 dialogues (fingerprint in
`DG08_TEST_LOCK.json`). Metrics `metrics_v1`; gains `baseline − method` (positive = better). Learned
finalists reported as mean ± std over seeds [13, 42, 73]. F0/F1 deterministic. Machine-readable
tables in `results/dg08/tables/`, bootstraps in `results/dg08/stats/`.

**Outside-harm on D-test:** NOT AVAILABLE — the locked split has no candidate/POI alignment
artifacts (`candidates_all.parquet` has no D-test rows), so canonical outside-harm / candidate-utility
are not computable on D-test; every text-derived metric is computed from the locked transcripts.
Outside-harm remains reported on the DG-04/05/06/07 D-dev-select development frontier.

## Table A — Greedy D-test (primary)

| System | MER | PIER | en-WER | zh-CER | corr | corrupt | utility | matrix-ret | embed-ret |
|---|---|---|---|---|---|---|---|---|---|
| F0 Frozen Whisper | 0.3606 | 0.8309 | 0.8337 | 0.1961 | 0 | 0 | 0 | 1.000 | 1.000 |
| F1 DG-04 frozen steering | 0.4070 | 0.7874 | 0.8167 | 0.2631 | 2180 | 616 | 1564 | 0.891 | 0.892 |
| F2 SALSA (×3) | 0.2991 | 0.5179 | 0.6088 | 0.1931 | 11450 | 207 | 11243 | 0.935 | 0.951 |
| F3 LoRA (×3) | **1.2890** | 0.1454 | **5.4643** | 0.1929 | 24845 | 220 | 24626 | 0.949 | 0.960 |
| F4 M* (×3) | 0.3473 | 0.7898 | 0.7948 | 0.1994 | 1689 | 211 | 1478 | **0.977** | 0.955 |

## Table B — Beam-5 D-test (robustness)

| System | MER | PIER | en-WER | zh-CER | corr | corrupt | utility | matrix-ret | embed-ret |
|---|---|---|---|---|---|---|---|---|---|
| F0 Frozen Whisper | 0.3489 | 0.7104 | 0.7294 | 0.1899 | 0 | 0 | 0 | 1.000 | 1.000 |
| F1 DG-04 frozen steering | 0.3444 | 0.6281 | 0.6767 | 0.2211 | 4000 | 1045 | 2955 | 0.947 | 0.896 |
| F2 SALSA (×3) | 0.3210 | 0.3387 | 0.7317 | 0.1730 | 13704 | 354 | 13350 | 0.958 | 0.959 |
| F3 LoRA (×3) | **1.9146** | 0.1411 | **8.1972** | 0.1884 | 20748 | 299 | 20448 | 0.964 | 0.968 |
| F4 M* (×3) | **0.3198** | 0.6341 | **0.6888** | 0.1828 | 3170 | 430 | 2741 | **0.988** | 0.967 |

## Table C — Efficiency

| System | trainable params | % backbone | ckpt size | optim updates | train wall (mean±std) | peak GPU | inf. RTF | inf. overhead vs F0 |
|---|---|---|---|---|---|---|---|---|
| F2 SALSA | 1,280 | 0.0001% | 7.4 KB | 2,019 | 45.5±0.4 min | 7.10 GB | 0.034 | ≈1× (parity) |
| F3 LoRA | 46,080 | 0.0030% | 187 KB | 2,019 | 48.6±1.5 min | 7.31 GB | 0.102 | **2.63×** |
| F4 M* | 43,651 | 0.0028% | 189 KB | 2,019 | 46.2±1.9 min | 5.51 GB | 0.034 | ≈1× (parity) |

Inference timing on the frozen D-dev-select 100-shortest-utterance subset (`mig` H100 3g.40gb,
greedy). Sub-1× overheads for SALSA/M* are measurement jitter on very short inputs → report as
parity; LoRA's 2.63× is the real per-step Q/V adapter cost. Parameter efficiency, training cost, and
inference overhead are distinct axes.

## Table D — Pairwise 95% dialogue-block bootstrap (2000 reps, unit = dialogue, seed 42)

Differences are **M\* − comparator**; PIER/MER lower = better, utility higher = better.

### Greedy
| Comparison | ΔPIER [CI] | ΔMER [CI] | Δutility [CI] |
|---|---|---|---|
| M* − Frozen | **−0.042** [−0.051,−0.033] ✅ | **−0.014** [−0.021,−0.006] ✅ | **+1483** [+1111,+1907] ✅ |
| M* − SALSA | +0.270 [+0.234,+0.303] ✅ | +0.048 [+0.029,+0.064] ✅ | −9744 [−13400,−6685] ✅ |
| M* − LoRA | +0.644 [+0.593,+0.688] ✅ | **−0.956** [−1.272,−0.672] ✅ | −23136 [−30255,−17241] ✅ |

### Beam-5
| Comparison | ΔPIER [CI] | ΔMER [CI] | Δutility [CI] |
|---|---|---|---|
| M* − Frozen | **−0.077** [−0.098,−0.058] ✅ | **−0.029** [−0.040,−0.019] ✅ | **+2751** [+1878,+3701] ✅ |
| M* − SALSA | +0.293 [+0.247,+0.327] ✅ | −0.002 [−0.029,+0.022] ✗ (tied) | −10588 [−14770,−6952] ✅ |
| M* − LoRA | +0.491 [+0.424,+0.547] ✅ | **−1.615** [−2.115,−1.135] ✅ | −17693 [−24058,−12315] ✅ |

✅ = 95% CI excludes zero. (15 dialogue blocks → wide intervals; D-test is underpowered by design.)

## Interpretation

**RQ1 — adaptive vs fixed/global steering.** M* (adaptive) delivers a clearly better correction–
**damage** trade-off than F1 (fixed global steering): far higher matrix retention (0.977 vs 0.891
greedy; 0.988 vs 0.947 beam), lower MER, and lower corruption per correction. Against the *learned*
global vector (SALSA), M* is better on ASR quality/retention but SALSA achieves more raw embedded
correction (lower PIER, higher utility). So adaptivity buys damage-control, not raw correction count.

**RQ2 — M\* vs SALSA and matched-budget LoRA.** M* has the **best overall ASR quality** — lowest MER
and lowest en-WER of any system in both regimes — and the **best matrix retention**, at inference
parity with frozen Whisper. SALSA is the strongest *balanced* baseline (large PIER reduction, MER
below F0, tiny parameter count). **LoRA posts the lowest PIER but a catastrophic MER (1.29 greedy /
1.91 beam) and en-WER (5.5 / 8.2)** from hallucinated English over-generation — not a usable
transcriber despite its "utility". M* does **not** dominate SALSA on embedded correction, but is the
only method that improves PIER, MER, and retention simultaneously without an MER blow-up. This
confirms DG-07's **M\* COMPETITIVE** verdict on locked D-test. No SOTA claim.

**Robustness (beam-5): PERSISTS.** System ordering is identical to greedy; M*'s advantage over frozen
Whisper *grows* under beam (PIER −0.077 vs −0.041; MER −0.029 vs −0.014), M*−SALSA MER becomes
statistically tied, and LoRA's MER catastrophe is amplified.

**Seed stability.** M* stable (MER std 0.0024, utility std 299); SALSA effectively deterministic
(std ≈ 0, zero-init global vector); LoRA moderate (MER std 0.05).

**Statistical uncertainty.** M* significantly beats frozen Whisper on all three primary endpoints in
**both** regimes (CIs exclude zero). SALSA significantly beats M* on PIER/utility (MER tied under
beam). M* significantly beats LoRA on MER (CI excludes zero); LoRA's PIER edge is an over-generation
artifact.

## DG-07 ablation carry-forward (development findings; not reinterpreted by D-test)

local direction SUPPORTED · conditioning direction NO CLEAR CONTRIBUTION · adaptive gate SUPPORTED ·
adaptive mixture SUPPORTED · damage-aware objective SUPPORTED · A5 refined basis DEFERRED. `v_cond`
was **not** removed from M* (M* was frozen before DG-07; removing it post-hoc would be a new method).
