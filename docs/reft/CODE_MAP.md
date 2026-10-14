# CS-ASR-ReFT code map (P0/P1)

| Component | Path | Notes |
|---|---|---|
| Config (frozen) | `configs/reft/p0p1.yaml` | model identity anchors, decode, data roles, P0 gate, P1 site/rank/policy/budgets, P1 gate |
| Paths, write-once I/O, reference firewall | `reft/common.py` | artifacts under `artifact_root` (never versioned); `audio_hash_ok` uses the manifest prefix-hash convention |
| Provenance and freezes | `reft/provenance.py` | file hashes, git, environment, Slurm, bf16 weight hash, `freeze` / `check_freeze` (amendments `<stage>_rK.json`) |
| Text, units, alignment, first divergence | `reft/text.py` | copied ACL-0 / TTLS primitives with source commits; ReFT extensions: strata, English error kinds, FD categories, token→unit map |
| Data audit | `reft/data_audit.py` | `p0`: ACL-0 dev150 identity/role/disjointness/seal checks; `p1`: training panel, dialogue-disjoint split, targets |
| P0 baseline | `reft/baseline_auto.py` | `decode` (GPU, firewall), `seal`, `verify-audio`, `evaluate` (verdict); `load_model` / `generate` are reused by P1 |
| Site | `reft/site.py` | post-cross-attention pre-FFN `h = q + s` of `configs p1.layer` (**31**, AMENDMENT L31); `delta` added to the cross-attention output |
| Gradient capture | `reft/gradient_capture.py` | `smoke` (acceptance), `capture` (Auto beam-5 decode, rows A/B/P, `q`, `h`, `g_s`) |
| Basis | `reft/build_basis.py` | frozen row policy, weights, float64 SVD, sign convention, eigh reconstruction, controls |
| Evaluation | `reft/evaluate_basis.py` | `leverage` (GPU oracle local diagnostics), `analyze` (Tests 1–4, bootstrap, verdict) |
| Tests | `tests/reft/` | CPU only: text/FD/kinds/sampling/write-once; tiny real Whisper for site identity, causality, autograd vs FD, batching; row rules; caps; controls |
| Slurm | `slurm/reft_cpu.sbatch`, `slurm/reft_p0.sbatch`, `slurm/reft_p1.sbatch` | MIG 3g.40gb for GPU steps; `TMPDIR` and logs under `/mnt/data/tungnx` |

## Artifact layout (local, `/mnt/data/tungnx/cs-asr-reft/`)

- `p0/`: P0 (unchanged by AMENDMENT L31).
- `p1/`: layer-independent shared inputs (`data_audit.json`, `panel.json`, `targets.json`, `decode.jsonl`), plus the
  **historical layer-24** capture and basis, which are untouched.
- `p1/l31/`: the active layer-31 smoke, gradients, basis, controls, leverage, preservation and evaluation (freeze stage
  `p1_l31`). Selected by `configs p1.artifact_subdir` through `reft.common.sdir`.

## Commands

```bash
python reft/provenance.py freeze --stage p0                # and --stage p1_l31 (active); amendments need --note
sbatch slurm/reft_cpu.sbatch test
sbatch slurm/reft_cpu.sbatch audit-p0
sbatch slurm/reft_p0.sbatch                                # Auto beam-5 decode
sbatch slurm/reft_cpu.sbatch p0-seal
sbatch slurm/reft_cpu.sbatch p0-verify-audio
sbatch slurm/reft_cpu.sbatch p0-evaluate
sbatch slurm/reft_cpu.sbatch audit-p1
sbatch slurm/reft_p1.sbatch smoke
sbatch slurm/reft_p1.sbatch capture
sbatch slurm/reft_cpu.sbatch basis
sbatch slurm/reft_p1.sbatch leverage
sbatch slurm/reft_cpu.sbatch p1-analyze
```

## Not implemented here

- The P2 controller `h + α·V·tanh(A·Vᵀ·Norm(h) + b)`.
- LoReFT and LoRA baselines.
- Free-decoding evaluation of any intervention.
