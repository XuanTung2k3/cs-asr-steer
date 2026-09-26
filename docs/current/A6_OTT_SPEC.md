# A6-OTT — Oracle Per-Sample Test-Time Steering Upper-Bound Search (SCIENTIFIC SPEC)

Status: **FROZEN PROTOCOL — compact upper-bound search. Supersedes the large BASIS-A6
execution.** A6-OTT replaces the 70,080-cell BASIS-A6 A6-F/A6-TT atlas *for execution* with a
small, phased, oracle **per-sample** search. The superseded atlas is preserved as evidence
(`results/basis_a6_expanded/`), not deleted. **No new scientific GPU jobs are submitted by the
freezing session.**

Companions: `A6_OTT_SELECTION_PROTOCOL.md` (phases + leakage firewall),
`A6_OTT_EXECUTION_PLAN.md` (superseded-job policy, reuse policy, compute). Migration record:
`results/a6_ott_upper_bound/migration/{SLURM_JOB_AUDIT.md,EXISTING_RESULT_REUSE_MANIFEST.json}`.
Prior frozen protocol carried over verbatim where noted:
`BASIS_A6_ORACLE_TT_SPEC.md` (A6-TT), `BASIS_A6_ASCEND_DATA_SPEC.md`.

---

## 0. Scope in one paragraph

A6-OTT includes **ONLY Oracle Per-Sample Test-Time Steering**. **No new corpus-level
fixed-vector experiments.** For each evaluation utterance `x`, the steering direction is built
**from `x` itself** using oracle **location** information only (never gold token identity),
applied Oracle-local to that same `x`, and decoded. It is an **UPPER BOUND**, not deployable
inference. The study runs in three phases — **Search → Confirm → Transfer** — with an explicit
data/selection **leakage firewall** (`A6_OTT_SELECTION_PROTOCOL.md`).

---

## 1. Methods (THREE families — FROZEN)

| Family | Construction | Side |
|---|---|---|
| **Add-Unique encoder** | `d = +v_unique^(x,l)` (per-sample) | encoder |
| **Add-Unique decoder** | `d = +v_unique^(x,l)` (per-sample) | decoder |
| **Conditioning-CS decoder** | `d = v_cond_cs^(x,l)` (per-sample paired `c_E`/`c_M`) | decoder |

Per-sample `v_unique^(x,l)` uses the A5 geometric extraction with dynamic rank (§4);
`v_cond_cs^(x,l)` uses paired analysis passes over oracle embedded-language positions on the
**baseline-hypothesis** sequence. Exact definitions are inherited **verbatim** from
`BASIS_A6_ORACLE_TT_SPEC.md` §4–§6 (Add-Unique = A6-TT `add_unique`; Conditioning-CS = A6-TT
`conditioning_cs`). **No Minus-Shared, no U-S, no Raw, no Conditioning-All, no fixed vectors**
in A6-OTT (they remain defined but out of this compact study's scope).

---

## 2. Models / datasets (FROZEN)

- **Models:** Whisper-large-v3 (enc E0–E31, dec D0–D31, d=1280); Qwen3-ASR-1.7B (enc E0–E23
  d=1024, dec D0–D27 d=2048).
- **Datasets:** CS-Dialogue (`D-dev-select`, 300), ASCEND (`ASCEND-eval`, realized **N=220**),
  SEAME-man (50), SEAME-sge (50). ASCEND-construct/`train` and all `test`/`D-dev-confirm`
  splits remain locked out per `BASIS_A6_ASCEND_DATA_SPEC.md` / `DATA_EXPOSURE.md`.

---

## 3. Oracle information budget (FROZEN — inherited)

Location-only, exactly as `BASIS_A6_ORACLE_TT_SPEC.md §2`: allowed = true CS/language-region
labels, accepted acoustic span alignment, reference↔hypothesis region alignment. **Forbidden =
gold correct token identity.** Directions are constructed on the **baseline hypothesis** (the
model's own free decode under the requested mode), never by teacher-forcing the gold reference.
Every A6-OTT direction record carries `gold_leakage=false`; any unavoidable exception is
documented per utterance.

---

## 4. Dynamic rank / eligibility (FROZEN — inherited)

Per `BASIS_A6_ORACLE_TT_SPEC.md §5,§7`: `r_(x,l) = min(32, n_A, n_B, d)`, no zero-pad,
truncated subspaces. Add-Unique requires the unique index to exist (`r_(x,l) ≥ 1` for a single
top direction; the shared/paired indices are not needed for Add-Unique). Conditioning-CS
requires ≥1 oracle embedded-language pooled position. Ineligible (utterance, layer) cells are
recorded `INELIGIBLE_*`, never silently substituted. Eligibility coverage (total/eligible/rate,
`n_A`,`n_B`,`r_x` median/p10/p90) is reported per (model, dataset, side, layer).

---

## 5. Localization / site / NormPreserve (FROZEN — inherited)

Oracle-local only (encoder = embedded-English acoustic frames; decoder = embedded-English
transcript positions), repaired Qwen local masks, frozen A4 sites, NormPreserve, no depth
rescale — all inherited from `BASIS_A6_ORACLE_TT_SPEC.md §1,§8` and A4/A5. **No Global.**

---

## 6. Dose + decoding (per phase — FROZEN)

Dose and decode are **narrowed per model/phase** vs the superseded atlas (that narrowing is the
point of the compact search):

- **Phase A (Search), greedy only:** Whisper `ρ ∈ {0.5,1,2}`; Qwen `ρ ∈ {0.5,1,2,4}`.
- **Phase B (Confirm):** the ≤6 selected settings/model, decoded under **both** `greedy` and
  `official_standard`.
- **Phase C (Transfer):** the one frozen setting/family/model, on SEAME, under `greedy` and
  `official_standard`.

Decoding semantics are the frozen A6 pair (`BASIS_A6_EXPANDED_SPEC §7`): Whisper
`official_standard = beam-5`; Qwen `official_standard ≡ greedy` (no beam in its shipped config).

---

## 7. Phase structure + logical evaluation counts (FROZEN)

Sites per model = (Add-Unique enc layers) + (Add-Unique dec layers) + (Conditioning-CS dec
layers): Whisper 32+32+32 = **96**; Qwen 24+28+28 = **80**.

| Phase | Data | Whisper logical | Qwen logical |
|---|---|---:|---:|
| **A — Search** | 20 CS + 20 ASCEND = 40 utt | 96 × 3 ρ × 40 = **11,520** | 80 × 4 ρ × 40 = **12,800** |
| **B — Confirm** | 280 CS + 200 ASCEND | ≤ **5,760** | ≤ **5,760** |
| **C — Transfer** | SEAME-man 50 + SEAME-sge 50 | ≤ **600** | ≤ **600** |

Phase-B/C caps are maxima (≤6 settings/model in B; one setting/family/model in C, ×2 decode ×
panels). A **conditional bounded Qwen rescue** (§8) may add a small decoder site/ρ screen on 10
CS + 10 ASCEND search utterances **only if** the Phase-A trigger fires — not broadened
automatically.

---

## 8. Conditional Qwen rescue (FROZEN — bounded)

Retained only if the Phase-A trigger fires (`A6_OTT_SELECTION_PROTOCOL.md §Rescue`). Scope:
10 CS + 10 ASCEND search utterances, a small Qwen **decoder** site/ρ screen. Do not broaden
automatically; any expansion beyond this bounded screen requires a new human decision.

---

## 9. Metrics (FROZEN — canonical + TT extras)

Canonical A4/A5 metrics + A6 diagnostics + A6-TT extras (eligibility coverage,
direction-construction latency, extra analysis passes, total inference latency/RTF), exactly as
`BASIS_A6_ORACLE_TT_SPEC.md §12`. `poi_net_utility` identity holds for every accepted row. U/S
common-subset machinery is not needed (Add-Unique + Cond-CS only), but eligibility coverage is
still reported per family.

---

## 10. Provenance (FROZEN — inherited)

Per-sample direction records (utterance_id, method, model, side, layer, decode analysis mode,
n_A, n_B, rank, direction_hash, oracle_alignment_hash) → `DIRECTION_BUNDLE_HASH` per
(model × dataset × side × family × decode mode). Every aggregate row references its bundle hash
(`BASIS_A6_ORACLE_TT_SPEC.md §11`). Existing compatible artifacts are reused under the reuse
policy (`A6_OTT_EXECUTION_PLAN.md`); the reuse manifest binds each to its hash.

---

## 11. Relationship to the superseded BASIS-A6

- **Superseded for execution:** the 70,080-cell A6-F/A6-TT atlas and its wave coordinator /
  atlas array jobs (cancelled; see `migration/SLURM_JOB_AUDIT.md`).
- **Preserved:** all of `results/basis_a6_expanded/` (baselines, partial oracle-TT caches,
  fixed inventories, ASCEND manifests, preflight). Nothing deleted.
- **Corpus-level fixed results are NEVER reused as oracle-TT** (reuse manifest classifies them
  `INVALID_OLD_RESULT`).

---

## 12. Headline scientific question (inherited)

Does per-sample (especially **Qwen**) steering become **substantially stronger** than
corpus-level steering? YES ⇒ strong input-dependent directions; NO (with real perturbation
magnitude) ⇒ evidence toward intervention-site / causal-pathway limitations. **Not forced
either way.** Nulls are valid, reportable outcomes.

---

## 13. Protocol Status: **PASS**

Scope (oracle per-sample only; 3 families), models, datasets, oracle budget, dynamic
rank/eligibility, localization, per-phase dose/decode, phase logical counts, conditional rescue,
metrics, provenance, and the superseded-atlas relationship are determined and frozen. Selection
firewall in `A6_OTT_SELECTION_PROTOCOL.md`; compute/reuse in `A6_OTT_EXECUTION_PLAN.md`. No new
scientific GPU jobs launched by the freezing session.
