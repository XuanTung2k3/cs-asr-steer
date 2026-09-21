# A6-OTT — Selection Protocol & Leakage Firewall (FROZEN)

Status: **FROZEN PROTOCOL.** Defines the outcome-blind Search/Confirm/Transfer split, the
selection rules (frozen **before** any Phase-A outcome is inspected), and the **critical**
data/selection leakage firewall. Companion to `A6_OTT_SPEC.md`, `A6_OTT_EXECUTION_PLAN.md`.

---

## 1. Split (deterministic, outcome-blind — FROZEN)

Realized eligible panels: CS `D-dev-select` = 300; `ASCEND-eval` = 220; SEAME-man = 50;
SEAME-sge = 50.

| Role | CS-Dialogue | ASCEND | SEAME |
|---|---|---|---|
| **SEARCH** (Phase A) | `CS_SEARCH_20` (20) | `ASCEND_SEARCH_20` (20) | — |
| **CONFIRM** (Phase B) | `CS_CONFIRM_280` (remaining 280) | `ASCEND_CONFIRM_200` (remaining 200) | — |
| **TRANSFER** (Phase C) | — | — | SEAME-man 50 + SEAME-sge 50 |

Rules:
- `CS_SEARCH_20 ⊂ D-dev-select`, `CS_CONFIRM_280 = D-dev-select \ CS_SEARCH_20`
  (disjoint, exhaustive, 20+280=300).
- `ASCEND_SEARCH_20 ⊂ ASCEND-eval`, `ASCEND_CONFIRM_200 = ASCEND-eval \ ASCEND_SEARCH_20`
  (20+200=220).
- **Search IDs are frozen OUTCOME-BLIND** (§2): selected deterministically from the frozen eval
  manifests (seed=42, stratified where practical), **before** any Phase-A metric is computed or
  read. Freeze `SEARCH_FREEZE.json` (exact ordered IDs + fingerprints + seed + rule hash) as the
  first Phase-A artifact.

---

## 2. Leakage firewall (CRITICAL — FROZEN)

This firewall is the core scientific-validity guarantee.

1. **Outcome-blind Search freeze.** `CS_SEARCH_20` / `ASCEND_SEARCH_20` are chosen from
   manifest metadata (IDs, stratification keys, fingerprints) **only** — never from any decoded
   outcome. The selection code path must not import/load any `*.jsonl` result metric, any
   `poi_*`, MER/PIER, retention, or utility value while freezing search IDs.
2. **No confirmation/SEAME outcomes during Phase-A selection.** During Phase-A candidate
   selection (§3), outcome metrics of `CS_CONFIRM_280`, `ASCEND_CONFIRM_200`, SEAME-man, and
   SEAME-sge artifacts **MUST NOT be loaded**. It **is allowed** to detect that compatible
   files exist via metadata / IDs / hashes / filenames / manifests (for reuse planning), but the
   selection code must not read their scientific outcome values.
3. **Phase ordering of reads.** Phase-B outcomes may be read/reused **only after** the Phase-A
   top-2 candidates are frozen. **SEAME (Phase-C) outcomes may be read only after the final
   per-family/model setting is frozen** at the end of Phase B.
4. **Reuse ≠ outcome read.** Reusing an existing per-sample *direction vector* (construction
   artifact) or an existing *baseline decode* is permitted where hashes match; reusing an
   existing *steered outcome metric* is governed by the phase ordering above.
5. **Implementation obligation.** The selection module enforces (1)–(4) in code (an allowlist of
   loadable fields at each phase); a violation is a hard failure, not a warning. This firewall is
   documented in `A6_OTT_EXECUTION_PLAN.md` (exact-result reuse policy) and audited in the final
   manifest.

---

## 3. Phase A → selection rule (FROZEN BEFORE inspecting Phase-A outcomes)

Three families × two models. **Select top 2 settings per (family, model)** on Phase-A Search
data (40 utt), by the pre-registered ranking rule below. The rule is frozen here, before any
Phase-A outcome is inspected.

- **Candidate setting** = (family, layer, ρ) at greedy, evaluated on the 40 Search utterances
  (Whisper ρ∈{0.5,1,2}; Qwen ρ∈{0.5,1,2,4}).
- **Ranking key (pre-registered, lexicographic):**
  1. **`poi_net_utility`** (corrections − corruptions) on the eligible Search subset — higher is
     better;
  2. **`PIER` gain** vs baseline — higher is better;
  3. **`matrix_retention`** — higher is better (tie-break, damage guard);
  4. **`intervention_energy`** — lower is better (final tie-break, efficiency).
- Eligibility guard: a setting is rankable only if its Search eligibility rate ≥ a pre-declared
  floor (default 0.5); settings below the floor are recorded but not selected.
- Output: `PHASE_A_SELECTION.json` — the top-2 (family, layer, ρ) per (family, model), the
  ranking table, and the frozen rule hash. Do **not** use confirmation or SEAME results.

Total Phase-B settings/model ≤ 3 families × 2 = **6** (matches `A6_OTT_SPEC §7`).

---

## 4. Phase B → Confirm (FROZEN)

- Data: `CS_CONFIRM_280` + `ASCEND_CONFIRM_200`.
- Settings: the ≤6 Phase-A winners/model.
- Decode: **greedy AND official_standard**.
- Max logical evaluations ≤ 5,760/model (`A6_OTT_SPEC §7`).
- Existing compatible Phase-B artifacts (by hash) may now be reused (post Phase-A freeze).
- Output: `PHASE_B_CONFIRM.json` with per-setting confirm metrics under both decode modes.
- **Freeze one setting per (family, model)** for Phase C by a pre-registered rule: max
  confirm-set `poi_net_utility` (greedy primary), ties broken by PIER gain, then matrix
  retention, then energy; the choice must be robust across both decode modes (if greedy and
  official_standard disagree on the winner, prefer the setting that is top-2 under both). Record
  `PHASE_B_FINAL_SETTINGS.json` + rule hash **before** reading any SEAME outcome.

---

## 5. Phase C → Transfer (FROZEN)

- After `PHASE_B_FINAL_SETTINGS.json` is frozen, evaluate/reuse the one setting per
  (family, model) on SEAME-man (50) + SEAME-sge (50) under greedy AND official_standard.
- Max logical evaluations ≤ 600/model.
- SEAME outcomes are read **only now**. SEAME is exploratory (n=50), transfer-only, never used
  to re-select or re-tune any A6-OTT setting.
- Output: `PHASE_C_TRANSFER.json`.

---

## 6. Conditional Qwen rescue (FROZEN trigger + bound)

- **Trigger (pre-registered):** fires iff, after Phase-A, **no** Qwen family has a selected
  setting with positive Search `poi_net_utility` at a non-trivial eligibility rate (≥ floor),
  i.e. Qwen per-sample steering shows no corrective signal in the standard Phase-A screen.
- **Bounded rescue scope:** 10 CS + 10 ASCEND Search utterances (a subset of the frozen
  SEARCH_20 sets), a small **Qwen decoder** site/ρ screen (a handful of decoder layers and the
  Qwen ρ grid). Purpose: distinguish "no input-dependent direction" from "screen too narrow."
- **Do not broaden automatically.** Any expansion beyond this bounded screen requires a new
  human decision. The rescue result is reported as diagnostic, feeding the headline
  input-dependent-vs-site-limitation question, not as a new selection surface.

---

## 7. Protocol Status: **PASS**

Split, outcome-blind Search freeze, the leakage firewall, the pre-registered Phase-A/B/C
selection rules, and the bounded conditional Qwen rescue trigger are determined and frozen. No
outcome value is read out of phase order.
