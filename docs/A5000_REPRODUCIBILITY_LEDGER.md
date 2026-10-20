# A5000 reproducibility ledger

Base scientific source 000b0629e55ea2e6b65812e829a085143f9bad2a; preserved bootstrap commit
ada154e3386bf5f2d1446ca7d96ed45db5f0d917. Development branch a5000/e0-e1-pure-ce.
No push/merge. Historical sources/results untouched. New active source/manifests are committed before freeze.
Freeze JSON records exact source commit/file hashes, config, model and versioned role-manifest SHA.
Artifacts, predictions, training references, checkpoints and logs stay outside Git under
/home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1. IDs and nontext membership metadata are versioned explicitly.

Local controls:

```bash
scripts/a5000/e1_control.sh launch
scripts/a5000/e1_control.sh status
scripts/a5000/e1_control.sh logs
scripts/a5000/e1_control.sh stop
scripts/a5000/e1_control.sh resume
```

Remote tmux session a5000-e1 survives SSH disconnects. RUN.lock prevents parallel duplicate campaigns.
process.json records the Python PID; exit.json records remote shell exit. Stop uses a STOP sentinel, saving
at optimizer boundaries; interrupted partial logical batches are replayed from the last completed checkpoint.
Resume retains sealed validation predictions and optimizer/RNG state. Finished result directories are immutable.
Six-hour aggregate accounting includes unsuccessful sessions in budget.json; exhaustion pauses resumably.
Engineering status is not scientific success. Do not claim completion before comparison.json and exit status.

Codex self-check is an internal check, not independent external scientific review. Check intended site,
parameter budget, frozen gradients, CE-only source path, common decoder/scorer, disjoint roles, no C150 fitting,
consistent native/cached semantics, no official test access and commit/manifest/checkpoint linkage.
