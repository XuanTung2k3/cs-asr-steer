# P2-2 amendment r1 — repair of the label-consistency checker for targets capped inside a UTF-8 character

**Status.** This is an engineering repair of one independent check. The first P2-2A preflight (job 59346,
`p2_2/preflight_59346.json`, kept unchanged) returned `P2_2_BLOCKED_ONPOLICY_COVERAGE`. Training never started. The
amendment is recorded in `p2_2/freeze_r1.json`, which chains to `freeze.json` and records the hash of the blocked
preflight.

**Unchanged:**
- event construction, targets, eligibility and normalization;
- λ, seeds, data, the loss and the training loop;
- the coverage floors, including **0 label mismatches**;
- every decision rule.

## 1. Failure

Every numeric coverage floor passed on both CE42 and KL42:

| Measure | CE42 | KL42 |
|---|---|---|
| Valid events | 796 | 794 |
| Utterances | 385 | 387 |
| Dialogues | 27 | 27 |
| Valid SB events | 252 | 245 |
| Non-identity-prefix events | 331 | 294 |

Every engineering gate also passed, including bitwise reproduction of KL_seed42 epoch 1 by the M2 loop at λ_OP = 0.

The block came only from `label_mismatches = 1` in each source. It is the **same event** in the B0, CE42 and KL42
rollouts. Its prefix is token-identical to B0's, so it is not on-policy specific.

## 2. Diagnosis (training data only)

**The event.** It is an R event (PRE category). The reference continuation is `IELTS IELTS 考试…`, and its standalone
tokenization is `" I" "EL" "TS" " I" "EL" "TS" "考" <first 2 bytes of 试>`. The cap of 8 tokens per event falls inside
the UTF-8 encoding of 试. The 8th token is a genuine Whisper byte-level BPE token: the start of the correct next
character.

**The target is correct.**
- It is exactly the canonical tokenization of the reference continuation at the correct location r.
- The label (r, scope) is right.

**The checker failed.**
- `location_label_consistent` strictly UTF-8-decodes prefix + target, and returns false on the incomplete trailing
  character.
- It handled a cap that cuts a Latin word, but not one that cuts a multi-byte character.
- Gold CE in P2-1/P2-1b supervises such byte-level tokens routinely.

**Conclusion.** This is a check-construction defect, not a wrong reference suffix or a mislabelled location.

## 3. Repair

**`reft/p2_2_alignment.location_label_consistent`**:
- Only when the target was capped (`truncated`), a trailing incomplete UTF-8 sequence of at most 3 bytes is split off.
- The decoded part must be the prefix's units followed by reference units r, r + 1, … exactly.
- The split-off bytes must begin the UTF-8 encoding of the next reference unit.
- The check is otherwise unchanged. It stays strict for uncapped targets.

**Tests.** `tests/reft/test_p2_2.py::test_label_check_cap_inside_utf8_character` covers:
- the accepted case;
- the uncapped broken character, rejected;
- a wrong partial character, rejected;
- a wrong unit before the cut, rejected.

**Freeze chain.** `reft/p2_2_train.py` adds the chain (`freeze-r1`, `check_freeze` walking
`freeze.json → freeze_r1.json`), and `slurm/reft_p2_2_cpu.sbatch` adds the step.

## 4. Rerun

The CPU tests, then `freeze-r1`, then the **complete** P2-2A GPU preflight again. It writes a new
`preflight_<job>.json`, and all gates are re-evaluated by the amended frozen code. Training proceeds only if that new
preflight returns `READY_FOR_P2_2_TRAINING`.

**Cost.** About 0.7 MIG slice-hours.
