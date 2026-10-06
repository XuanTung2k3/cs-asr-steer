"""P2-DIR Exp-3 BROAD control: frozen per-utterance squared-energy budget (spec section 8).

BROAD (g = 1, selected direction) matches NEW's total realized squared edit energy per utterance:
``Q_u = sum ||edit_NEW||^2`` (sealed from the completed NEW decode, reference-free) and
``N_u`` = eligible UTF8-complete content queries (t >= 1) of B0's cached no-edit decode. Packets of
equal squared energy ``Q_u / N_u`` (chord norm ``sqrt(Q_u / N_u)``) are allocated sequentially at
BROAD's own eligible queries; the last packet is capped by the remaining budget; the budget is
decremented by the ACTUAL realized squared energy and clamped at zero after a rounding overshoot;
once exhausted every later edit is zero. Direction failure / unreachable consumes no budget. This
is a diagnostic control, not a deployable online algorithm.
"""
from __future__ import annotations

import math

TOLERANCE = 0.02


class BroadBudget:
    def __init__(self, q_new: float, n_eligible: int, tolerance: float = TOLERANCE):
        self.Q = float(q_new)
        self.N = int(n_eligible)
        self.tolerance = float(tolerance)
        positive = self.Q > 0 and self.N > 0
        self.packet = math.sqrt(self.Q / self.N) if positive else 0.0
        self.remaining = self.Q if positive else 0.0
        self.spent = 0.0
        self.packets = 0
        self.overshoot = 0.0

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def next_target(self) -> float:
        """Chord-norm target for the next eligible query (0 once exhausted)."""
        if self.exhausted:
            return 0.0
        return min(self.packet, math.sqrt(self.remaining))

    def consume(self, realized_norm: float) -> None:
        """Decrement by the actual realized squared energy of an applied edit."""
        sq = float(realized_norm) ** 2
        if sq <= 0.0:
            return
        self.spent += sq
        self.packets += 1
        self.remaining -= sq
        if self.remaining < 0.0:
            self.overshoot = max(self.overshoot, -self.remaining)
            self.remaining = 0.0

    def record(self) -> dict:
        return {"Q_new": self.Q, "N_eligible": self.N, "packet_norm": self.packet, "spent": self.spent,
                "remaining": self.remaining, "packets": self.packets, "overshoot": self.overshoot,
                "relative_mismatch": None if self.Q <= 0 else self.spent / self.Q - 1.0}


def energy_match(per_utterance: list[tuple[float, float]], tolerance: float = TOLERANCE) -> dict:
    """[(Q_new, Q_broad)] -> frozen BROAD validity: aggregate within 2% over positive-budget
    utterances AND at most 5% of positive-budget utterances individually mismatched by > 2%."""
    pos = [(q, b) for q, b in per_utterance if q > 0]
    if not pos:
        return {"valid": False, "reason": "no_positive_budget", "aggregate_mismatch": None, "unmatched": []}
    agg = sum(b for _, b in pos) / sum(q for q, _ in pos) - 1.0
    unmatched = [i for i, (q, b) in enumerate(per_utterance) if q > 0 and abs(b / q - 1.0) > tolerance]
    frac = len(unmatched) / len(pos)
    return {"valid": abs(agg) <= tolerance and frac <= 0.05, "aggregate_mismatch": agg,
            "unmatched": unmatched, "unmatched_fraction": frac, "positive_budget_utterances": len(pos)}
