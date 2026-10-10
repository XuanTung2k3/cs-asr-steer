# P2-SEQ pre-run audit — **PASS_TO_P2_SEQ**

Contract frozen at `660a619` (spec/design/config byte-unchanged). Start: local HEAD = origin/cs-asr-steer-inf =
`660a6197ca2ea49f6df296e92a751800f82e240e`, clean, empty queue. Implementation `ed9c355`; reuse proof sealed separately.
Machine verdict `results/inference_cf/p2seq/prerun_audit.json` (CPU only, no outcome).

Verified:
- all 20 config anchors;
- the exact panel: byte SHA256 266ea7ea…, saved order, 100 unique IDs, 20 dialogues × 5;
- the reuse seal is committed, self-hashed and pre-outcome;
- all 100 sealed historical P2-A L16 rows (hash/identity/status, B0_AUTO present) plus audio identity;
- every historical P2 source file is unchanged;
- the environment is identical (Python 3.11.9, torch 2.10.0+cu128, transformers 4.57.6) in the P0-R2 manifest and
  now; torch/transformers/tokenizers dist-info and every model file predate the historical run (2026-09-25);
- the AUTO reuse decision is consistent with these checks;
- no `run*` output exists;
- the runner's decode path contains no reference, evaluator or adaptation identifier;
- alpha is restricted to {0, 2}, with L16 and a 200-token cap;
- the auditor is independent, and the tests are present.

**Reuse decision (engineering-only, sealed before outcomes): `REUSE_AUTO_ALL_100`.** S0 (matched forced, alpha 0)
and S2 (STEER) are decoded newly through the single additive D2 sequence driver in one allocation; historical B0M is
not used as S0. Expected 200 new system-utterance passes. Reused AUTO runtime/VRAM is reported as unavailable.

Recorded engineering correction (pre-outcome): the first reuse proof (`reuse_audit_attempt1_hashdef_bug.json`,
preserved) compared a raw full-file SHA256 against the panel `audio_sha256`. The project's role-manifest audio
identity is `csasr.utils.hashing.sha256_file` (size-prefixed, first `audio_hash_bytes` = 65,536 bytes per
`configs/data/cs_dialogue.yaml`). That spurious mismatch would have forced a needless AUTO recomputation. The
fixed proof uses the frozen fingerprint plus file mtime < historical run start, and matches 100/100. The runner's
runtime audio check uses the same fingerprint. Test added.

Tests: `tests/test_inference_cf_p2seq.py` (11) plus the cached, P2, P2-DIR directions, P2-SEL, DG-02 site, lss sites
and canonical metric suites (93), all passing. Firewall: exposed D-dev-select 100 only. No TTA. P3 HELD.
