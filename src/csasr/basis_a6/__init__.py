"""CPU-testable implementation primitives for frozen BASIS-A6.

The package contains protocol accounting, ASCEND preparation, direction
construction, per-sample provenance/cache logic, and acceptance guards. Model
execution is intentionally kept in runners so importing this package never
loads a checkpoint or starts a GPU job.
"""

from .protocol import enumerate_a6_f, enumerate_a6_tt, expected_counts

__all__ = ["enumerate_a6_f", "enumerate_a6_tt", "expected_counts"]
