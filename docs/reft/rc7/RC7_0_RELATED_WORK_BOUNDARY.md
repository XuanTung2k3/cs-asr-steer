# RC7-0D — Related-work boundary

Verification: titles, venues and abstracts were checked by web search on 2026-10-26. Full texts were not read, so method details
beyond the abstracts are not asserted.

| Work | What it already establishes | Consequence for this project |
|---|---|---|
| **ReFT** (Wu et al., NeurIPS 2024; arXiv:2404.03592) | Defines LoReFT `h + Rᵀ(Wh + b − Rh)` and DiReFT `h + B(Ah + b)` as low-rank representation interventions. | The A5000 "E-ReFT" and "LRA" are these operators placed at Whisper's post-final-LayerNorm encoder output. Placement and the ASR evaluation are ours; the operators are not. Never call LRA a new adapter. |
| **Grft** (Zeng et al., ACL 2025 long, 2025.acl-long.506; "Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach") | A learned gate on top of ReFT-style low-rank representation edits, deciding per input how strongly to intervene (≈4.1K extra gate parameters). | Input-conditional gating of a ReFT edit is prior art. A gated LoReFT/DiReFT is not novel as a mechanism. |
| **TARE** (Wang & Zhao, ACL 2026 long, 2026.acl-long.841; "Lightweight Token-Aware Representation Editing") | Token-specific representation edits: a selector routes each token to a few lightweight diagonal editors after FFN blocks; compared against LoReFT and RED. | Per-position (token-wise) conditional representation editing is prior art. Moving from tokens to 1 s acoustic blocks is an application difference, not a new principle. |
| **GateRA** (Ou et al., AAAI 2026; arXiv:2511.17582; "Token-Aware Modulation for Parameter-Efficient Fine-Tuning") | A per-token sigmoid gate scales the PEFT update, with an entropy regularizer for near-binary gating. | Token-wise modulation of update *strength*, the closest analogue to energy-preserving reallocation, is prior art. |
| Internal H100 S0/S1 (`docs/reft/ARCH_S0S1_RESULT.md`) | Cross-attention-conditioned rank-2 LoReFT (M1) vs parameter-matched rank-3 LoReFT: `ARCH_S1_NO_ARCHITECTURAL_GAIN`. | Conditioning the edit on context already failed internally at the decoder site. |
| Internal A5000 N2 and Gated-LRA sprint | Gated LoReFT (G3–G5) and gated DiReFT (L2–L4), energy-normalized, audio-side features: `N2_NO_GO`, `NO_MEANINGFUL_L4_BENEFIT`. | The specific temporal energy-reallocation mechanism has been screened twice on this data. |

## Claims that remain open to this project

- **C1 (primary):** controlled evidence on compact *encoder-output* representation adaptation for code-switching ASR
  versus weight-LoRA. The evidence covers recognition, preservation (non-CS, looping) and measured cost. This is an
  empirical contribution and does not depend on novelty of the operator.
- **C3:** functional explanation: where corrections and damage occur, edit magnitude/placement, and what D1/D2
  readouts do and do not show.
- **C2 (conditional):** only as a *negative or bounded* result ("temporal energy reallocation of a compact
  encoder edit does not help beyond 0.5 pp under matched controls"). A positive C2 would need a hypothesis
  distinct from Grft/TARE/GateRA *and* from the archived G5/L4. RC7-0 did not find one.

## Claims to avoid

- Novelty of conditional, gated, token- or time-specific representation editing per se, including because it is
  applied to Whisper or to audio frames.
- "LoReFT/DiReFT beat LoRA": on V300, LoRA-Strong is best on every recognition axis. Only parameter-matched
  LoRA-PM is worse, and the selection is single-seed and selection-biased.
- Proportional memory or latency savings from parameter counts (see `RC7_0_EFFICIENCY_ACCOUNTING.md`).
