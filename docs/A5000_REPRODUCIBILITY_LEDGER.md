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

Primary experiment source: 61a380ff6d41c1d76005aa75bc1edf501b63366c.
Primary freeze SHA256: fdeea9195a0c0b37953674f518096224c79b8ea206212c5a9a7d072c52b70f91.
Role-manifest SHA256: ad43487483ab70fb717590cf1e7e805b771dce822cfad3bde4db7746f5c8eb2b.
B0 predictions SHA256: 8aec79d61636a920f2760145844ad12d9926e3ecd0fcf0c8eeebb0f4e0762641.
B0 seal retained across the actual graceful stop/resume exercise. PID initially3031, resumed3516;
control status reports the live process rather than assuming those PIDs remain active.
Follow-up code is separated from the frozen primary source; documentation/follow-up commits do not amend the
primary methods, recipe, cache, checkpoint selection or scorer. No objective GPU sweep was started.

Read-only result aggregation, paired descriptive dialogue bootstrap, versioned input audits and explicit
post-selection confirmation tooling are in reft/a5000_followup/{summarize,audit_inputs,selfcheck,confirm}.py.
Confirmation refuses unfinished primary arms or changed selected checkpoint hashes; source/selection is
frozen before opening the C150 model-evaluation reader. No duplicate model or second GPU is required.
Its separate process/status/exit files live under confirmation/; the primary RUN.lock also guards it.
Do not start it until the primary tmux session exits and the selected GPU is free.

The full-primary suite ran once at source61a380f: 1005 passed, 3 skips. Both real-tokenizer skipped tests then
passed at the verified portable model path. Follow-up suite: 23 passed, including nine objective paths,
exact optimizer-resume tests, D1/D2 checks and confirmation access/hash gates. Effective unique coverage:
1030 passed, one private historical-allocation check skipped/blocked, zero outstanding failures.
A first resume-unit fixture used unmatched GLOBAL Householder RNG states; the fixture was corrected to
match the already-seeded primary runner and both resume tests passed. No primary scientific code change.
