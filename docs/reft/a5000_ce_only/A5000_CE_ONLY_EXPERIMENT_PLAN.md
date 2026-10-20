# Requested protocol — not admitted

The owner requires historical Train394 / Inner106 / Full500 / Dev150. Exact identities were not recovered
in the preceding bounded inventory (docs/A5000_E0_SCIENTIFIC_SETUP.md). The current local bounded inventory
of Code/Draft and Downloads also found no matching split artifacts. The remote recheck was attempted but
SSH to the configured gateway timed out; this is not evidence of absence on an unreachable server.

Available configs/reft/a5000_manifests/roles_v1.json contains A5000_T394/V106/C150 with SHA256
ad43487483ab70fb717590cf1e7e805b771dce822cfad3bde4db7746f5c8eb2b.
These new identities cannot be relabeled as historical roles. C150 is not the exposed historical Dev150.
Full500 cannot be silently constructed as a historical population from the new roles.

Admission needs the exact four role manifests with source provenance and pinned hashes, utterance,
dialogue, speaker, original-partition and audio-hash membership. Full500 must equal Train394 union
Inner106; the training/selection components must be dialogue-disjoint; Dev150 must be distinct.
If recovered manifest schemas differ, add a lossless, provenance-bound metadata adapter. Do not resample.
No locked test manifest/audio needs to be opened.

The default config is JSON-compatible YAML and deliberately pins no absent role identities. The 24 GPU-hour,
36 wall-hour and 100 GiB caps are PROVISIONAL, not throughput-admitted launch settings. Mandatory encoder
LoRA feasibility and complete fair matrix coverage must be established before freezing a scientific plan.

A second change from the old pilot is valid-frame-only intervention. The old pilot adapts all 1500 frames;
its checkpoints cannot silently serve as the matched baseline of the new valid-frame-only comparison.
No GO thresholds have been frozen or tuned on gate results; no gate results exist.
