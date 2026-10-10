# Frozen MAP panel-selection algorithm

This algorithm is frozen before adapted outputs/reference outcomes are inspected. The actual24 IDs are already sealed in `P2_TTA_MAP_PANEL24.json`; do not reseat them after R. Selection does not authorize MAP: audited R NO_VIABLE is still required.

Inputs: exact byte-hash parent P2_SEL_MINI_PANEL.json and all100 hash-anchored theta0 FORCED/AUTO rows in master config. Audit model/audio/tokenizer/generation/execution semantics against original P2SEQ reuse proof first. Parent order is JSON `rows` order. Define D by exact stored decoded text inequality, with no normalization, labels or reference access; A by equality. The decoded text is the same historical transcript field, not generated tokens including language prompt. Candidate token differences may be logged later but cannot change D/A membership.

1. Group D rows by dialogue. Dialogue order is first appearance in the parent; each queue is parent order. Round-robin queues, one row per nonempty dialogue per round, until min(16,total D) rows selected.
2. Fill to24 from unselected A. At each step choose the available row minimizing `(number already selected in its dialogue, parent_index, dialogue_id, utterance_id)`. Increment dialogue count and repeat. This first prefers unseen dialogues, then least represented dialogues. There is no duration/error/transcript quality sort.
3. If A cannot fill24, INVALID before execution; no D overfill or other fallback. Reject duplicates or IDs outside parent. Seal ordered rows with group, parent index, dialogue and baseline hashes. No refilling after invalid/missing output.
4. Set LOW_DISAGREEMENT exactly if total parent D<6, with the interpretation frozen in master spec. Actual12 D /88 A, D spans9 dialogues. Selected12 D +12 A,20 dialogues.

Hash convention: SHA256 of exact UTF-8 JSON bytes including final newline, not of reserialized parsed data. Master config freezes that byte hash. Independent auditor replays the selection from100 raw theta0 rows without importing the primary selector. Original parent panel remains immutable.
