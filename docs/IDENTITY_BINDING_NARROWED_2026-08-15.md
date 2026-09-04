# Artifact identity narrowed: code/config gated, tests recorded

**Date:** 2026-08-15
**Change:** `identity.source_sha256` (one combined digest) is superseded, for
newly published artifacts, by `identity.code_config_sha256` (gated) plus
`identity.test_sha256` (recorded, not gated).
**Status:** Implemented and CPU-verified. No artifact was published, re-verified,
migrated, or modified. Production Gate A, every threshold, and every scientific
parameter are unchanged.

---

## 1. Why — three occurrences

`manifest.identity()` bound every artifact to `source_snapshot_hash()`, which
covers `src/`, `configs/`, **and `tests/`**. A test-only edit therefore
invalidated the identity of data artifacts it could not possibly have affected.
This did not stay theoretical:

1. **Job 40373 refused in 2 seconds.** The v2r2 role manifests and freeze were
   published at snapshot `d2812973…`; six regression tests were added minutes
   later, moving the tree to `82c96084…`. Every v2r2 artifact then failed
   execution-time identity with
   `identity_mismatch differs in ['source_sha256']`, and the first real GPU run
   against them refused before loading a model.
2. **The v2r2 → v2r3 republication.** Recovering from (1) required republishing
   an entire namespace — dialogue manifest, seven role manifests, and the spec
   freeze — solely to re-bind identical bytes to a new snapshot. The allocation
   was verified bit-identical dialogue-by-dialogue; nothing scientific changed.
3. **Session 8 hit it again.** The v2r3 artifacts were published at `64346ca6…`;
   the baseline driver, its tests, and one read-only diagnostic module have since
   moved the tree. Nothing in that session depended on identity verification, so
   it passed unnoticed — but the next run that does would refuse exactly as (1)
   did.

The failure mode is conservative and loud, which is why it never corrupted
anything. It is also expensive and, for `tests/`, unnecessary.

---

## 2. The split, and its exact file-sets

Three functions now exist in `csasr.utils.provenance`, all sharing one
`_snapshot_hash(repo_root, roots, files)` implementation:

| Function | Covers | Role |
|---|---|---|
| `code_config_snapshot_hash()` | every regular file under `src/` and `configs/`, plus `cs_asr_e1_e5.sh`, `cs_asr_nat5h.sh`, `cs_asr_lss.sh`, `pyproject.toml`, `environment.lock.txt` | **gated** |
| `test_snapshot_hash()` | every regular file under `tests/` | **recorded, never gated** |
| `source_snapshot_hash()` | `src/`, `tests/`, `configs/` then the root files, in that original order | legacy, unchanged |

All three exclude any path containing `__pycache__`.

Measured on the tree at the time of writing:

```text
code_config_sha256 = f7d82033825b94fbf61bc8f0dd6bd9bce70f7202919afcf1531d4145d4026c0d
test_sha256        = 6796e066d18235ed8dad56aa603bc059a7f990e90f433d3ee65e1e3f6572d5c0
legacy source_sha256 = 48c523962879f81898891ce191f1f41c2b1a1c7f838e1080b954097d634b8312
```

The two new sets **partition** the legacy set exactly:

| Set | Files |
|---|---:|
| code/config | 141 |
| tests | 43 |
| legacy | 184 |
| intersection | **0** |
| union vs legacy | **identical** — nothing missing, nothing extra |

Launchers, `pyproject.toml`, and `environment.lock.txt` stay on the gated side:
they define the environment a run executed in, so they belong with code.

---

## 3. Grandfathering

`verify()` recognises two **mutually exclusive** formats and refuses anything
else:

| Recorded fields | Branch | Gated on |
|---|---|---|
| exactly `source_sha256` | legacy | `source_sha256`, `config_sha256`, `data_sha256`, `model_id`, `model_revision` |
| exactly `code_config_sha256` + `test_sha256` | split | `code_config_sha256`, `config_sha256`, `data_sha256`, `model_id`, `model_revision` |
| neither, partial, mixed, or a non-digest value | **refused** | — `identity_mismatch: malformed source identity: …` |

`_legacy_identity()` is the pre-split `identity()` verbatim, so a legacy artifact
is compared against exactly the key set it was compared against before this
change. **No existing artifact is migrated, rewritten, recomputed, or
re-authenticated.**

An inventory of existing identity-bearing sidecars — metadata only, no
verification — found 110, all legacy:

| Root | Sidecars |
|---|---:|
| `artifacts_lss` | 28 |
| `artifacts_dialogue_v2r3` | 23 |
| `artifacts_dialogue_v2` | 14 |
| `artifacts_dialogue_v2r2` | 14 |
| `artifacts_dialogue_v1` | 13 |
| `diagnostics` | 18 |

Five further `*.manifest.json` files are generation-completion records rather
than per-artifact sidecars and carry no identity block: three role-generation
records, the v2r3 candidate-generation record, and the v2r3 baseline/POI record.

### What grandfathering does *not* do

Existing artifacts keep the old semantics **including the old flaw**. The 23
v2r3 sidecars still gate on `source_sha256`, so a test-only edit still
invalidates them. Verified directly: all 23 classify as `legacy`, all 23 pass
byte-level verification, and the sample comparison returns
`identity_mismatch: differs in ['source_sha256']` — the same result the pre-split
code produced. **Only artifacts published from now on benefit.** Migrating the
existing ones would mean rewriting immutable manifests, which is precisely what
the immutability rule forbids.

---

## 4. The guard, and the hole it closes

Excluding `tests/` from the gate weakens one real claim: that an artifact was
produced by exactly the tree present. The specific danger is logic migrating
into a `tests/` helper that production imports — then test-tree content would
affect scientific bytes while being ungated.

`tests/test_source_identity.py:174` closes that hole with an AST-based guard. It
parses `ast.Import` and `ast.ImportFrom` (resolving relative imports against the
containing package), builds the internal `src/` module dependency graph,
traverses it **transitively**, and reports complete offending chains naming the
source module and the imported `tests.*` module.

It is proven to detect, not merely asserted to: a synthetic fixture with
`demo.entry → demo.helper → tests.fixtures` is checked to produce **both**
`demo.helper -> tests.fixtures` and the full transitive chain
`demo.entry -> demo.helper -> tests.fixtures`. The guard passes on the real
repository.

### Residual limitations, stated

- The guard is static. A dynamic `importlib.import_module("tests.…")` would
  evade it. Every dynamic import in `src/` today resolves a third-party or
  config-named module — and `configs/` is gated — but this is a limitation, not
  a proof of impossibility.
- The guard covers imports, not file reads. Production opening a data file under
  `tests/` by path would be equally invisible to it. A scan for `tests/` path
  literals in `src/` currently finds none.

---

## 5. The trade accepted

**A test-only edit no longer refuses execution for artifacts published under the
split format.** That is the entire point, and it is a real loss of strictness.

It is acceptable because of what remains gated and what now guards the gap:

- Everything that can change an artifact's bytes — `src/`, `configs/`, the
  launchers, the environment lock — is still gated, byte for byte.
- `test_sha256` is still **recorded** in every new manifest, so the test tree at
  publication time remains fully reconstructible. Nothing is lost from the
  provenance record; only the gate narrowed.
- The one plausible route by which `tests/` could influence production bytes —
  an import — is now a failing test with transitive detection.
- The failure it removes was not protecting anything. In all three occurrences
  the refusal was triggered by added regression tests, and the correct response
  was always to re-bind identical bytes.

The asymmetry was demonstrated on a scratch copy of the tree rather than
asserted:

| Edit | `code_config_sha256` | `test_sha256` | legacy |
|---|---|---|---|
| baseline | `f7d82033…` | `6796e066…` | `48c52396…` |
| add a file under `tests/` | **unchanged** | changed | changed |
| add a file under `src/` | **changed** | unchanged | changed |
| add a file under `configs/` | **changed** | — | — |

---

## 6. Review findings and resolution

The supplied report contained **no BLOCKER, RISK, or SCOPE findings**. It is a
completion report describing an implementation already present in the tree, so
Phase 1 had nothing to revert and no disagreement to state. Rather than treat
that as vacuous, the implementation was reviewed independently here. Results:

| Check | Finding |
|---|---|
| Does anything outside `manifest.py` read `identity["source_sha256"]`? | **No.** The only occurrences are inside `manifest.py` itself and one config comment. The split cannot break an external consumer. |
| Does `nat5h_pipeline` / `provenance.run_provenance` still use the combined hash? | Yes, as a **record**, not a gate. Behaviour unchanged. |
| Is `_legacy_identity()` really the pre-split `identity()`? | Yes, verbatim — confirmed against the diff. |
| Is the legacy/split classification exhaustive? | Yes: legacy-only, split-only, and every other combination refused, with digest validation on both branches. |
| Can `tests/` reach production by a non-import route? | No `tests/` path literal appears in `src/`. Dynamic-import route noted as a residual limitation (§4). |

No code was changed in this session. Two documentation files were written.

---

## 7. Verification

| # | Check | Result |
|---|---|---|
| 1 | Compute both hashes on the current tree | `code_config_sha256 = f7d82033…`, `test_sha256 = 6796e066…` |
| 2 | File-sets exhaustive and disjoint vs legacy | 141 + 43 = 184; intersection **0**; union **identical** to legacy |
| 3 | v2r3 artifacts under the legacy branch | 23/23 classified `legacy`; 23/23 pass byte verification; identity comparison uses the pre-split key set and returns the pre-split result |
| 4 | Asymmetry on a scratch copy | `tests/` edit leaves `code_config_sha256` fixed; `src/` and `configs/` edits change it (table in §5). Scratch copy deleted; the real tree was never modified |
| 5 | `src/`-imports-`tests/` guard | passes on the current tree, and its synthetic fixture proves direct and transitive detection |
| 6 | No artifact modified | 0 files modified under any artifact root in the last 90 minutes; newest mtimes: `artifacts_lss` 2026-08-12 02:41:33, `artifacts_v2` 2026-07-28 10:46:20, `artifacts_dialogue_v2` 2026-08-14 17:00:10, `v2r2` 2026-08-14 19:41:13, `v2r3` 2026-08-15 04:15:06; exposure ledger `1786502490` |

CPU suite: see §9. `compileall` clean.

---

## 8. Consequences for the code freeze

Sessions that publish artifacts have been freezing `src/`, `configs/`, **and**
`tests/` between publication and job completion. For artifacts published under
the split format that is no longer necessary: the freeze needs to cover `src/`
and `configs/` (with the launchers and lock file) only.

Two caveats:

- Artifacts already published in the **legacy** format — every artifact that
  exists today — still gate on the combined hash. Any session consuming them
  under `require_identity=True` must still hold `tests/` still.
- Adding a test that a production module then imports would be caught by the
  guard as a test failure, not by the identity gate.

---

## 9. Open items

1. The 110 existing legacy sidecars are not migrated and will not be. They keep
   the old semantics permanently; the benefit begins with the next publication.
2. The guard is static and covers imports only (§4 limitations).
3. No artifact has yet been published under the split format, so the split
   branch of `verify()` has been exercised only by the regression suite, never
   against a real artifact on disk.

**No artifact was published or re-verified. No artifact's contents changed. No
gate was evaluated and production Gate A is unchanged.**
