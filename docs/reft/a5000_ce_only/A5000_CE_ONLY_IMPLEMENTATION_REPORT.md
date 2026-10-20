# CE-only temporal allocation — admission blocked

Status: A5000_CE_ONLY_PREFLIGHT_BLOCKED. This is not an implemented or launched R0–R6 campaign.

Implemented only: isolated admission CLI, exact manifest hash checks before parsing, four required role
identities/counts, utterance/audio uniqueness, train/selection dialogue and speaker disjointness,
Full500 exact metadata union, Full500/Dev150 separation, CE-only/test firewall checks, immutable preparation
records, status/report commands, and explicit launch/resume refusal while the executor is absent.
Five focused standard-library CPU tests pass. Synthetic metadata fixtures are tests, never scientific evidence.

Not implemented: temporal gate, full training/decode integration, encoder weight-LoRA, R0–R6 controller,
retry/recovery, resource admission, pilot criteria, conditional confirmation or plots. No GPU preflight
was run for this new method. Existing E1 GPU results are not admission evidence for the new method.

Current local source is preserved on exp/a5000-ce-only-adaptive-reft. No legacy scientific source is changed.
