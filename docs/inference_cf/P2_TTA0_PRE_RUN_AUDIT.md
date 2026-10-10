# P2-TTA0 — pre-run audit

**Verdict: `PASS_TO_P2_TTA0`** (`results/inference_cf/p2tta0/prerun_audit.json`, 33/33 checks). Run before any
adaptation outcome, against freeze `eb0da2a` and implementation `503c874`.

| Check | Result |
|---|---|
| Spec/design/config/panel20 byte-identical to freeze; 17 anchor hashes | pass |
| Panel20 = first row per dialogue of the parent 100 panel, 20 IDs / 20 dialogues, sha256 `3c752d7a…` | pass |
| Trainables: actual decoder `nn.LayerNorm` affines on the real config (meta device) = frozen 194 names/shapes, 248,320 of 1,543,490,560 scalars (0.0160882%), no encoder tensor | pass |
| Optimizer: AdamW lr 1e-3, wd 0, betas (0.9, 0.999), eps 1e-8, no amsgrad/foreach/fused/maximize, 2 steps; fp32 masters via `functional_call(..., strict=False, tie_weights=True)`, `use_cache=False` | pass |
| Pseudo seal committed before adaptation; `outcomes_computed=false`, `references_used=false`; all reuse-proof checks true | pass |
| Teachers: y_B = P2-SEQ S0 tokens, y_A = historical AUTO tokens, terminations, valid masks recomputed independently, full audio sha256 | pass |
| Suppression/EOS = generation config; P2-SEQ audit PASS and AUTO reuse proof | pass |
| Environment (python 3.11.9, torch 2.10.0+cu128, transformers 4.57.6); no run outputs yet | pass |
| Runner/module free of references, evaluator, D2, LID, edit hook, sampling, beams, clipping, scheduler, scaler, autocast | pass |
| Auditor imports neither the analysis nor `episodic_tta`; focused tests present (14 pass) | pass |

Reuse: B0-FORCED = sealed P2-SEQ S0 (the run re-decodes theta0 and must reproduce it); B0-AUTO = historical AUTO
under the P2-SEQ proof. Audio identity is recorded under two labels, `audio_fingerprint_64k_sizeprefixed` and
`audio_full_sha256`. No AUTO is regenerated.
