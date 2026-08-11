"""Localize-Select-Steer: the selective test-time intervention pipeline.

Library code for the ARR-October-2026 method paper (see
`docs/proposal_arr/`). Stage entry points live in `csasr.experiments.lss_*`;
everything here is importable without a GPU and without loading a model.

The package deliberately does not modify `csasr.nat5h`, whose artifacts must
stay reproducible: alignment machinery is wrapped, never edited.
"""
