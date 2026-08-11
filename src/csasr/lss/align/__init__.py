"""Alignment diagnostics and validation for the LSS pipeline.

`csasr.nat5h` is treated as read-only: its consensus machinery and schema are
reused, never edited, so the exploratory artifacts it produced stay
reproducible. Defects in it (a rejection code hard-coded to `_gt_200ms`, a
confidence bin fixed at 100 ms) are corrected by wrapping.
"""
