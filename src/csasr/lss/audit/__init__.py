"""Blinded human boundary audit: the only external, natural ground truth.

Synthetic splices give real error but on concatenated speech, which has no
cross-boundary coarticulation and no natural switch prosody, so they are an
optimistic bound by construction. Human verdicts on natural audio are what
Gate A ultimately rests on.
"""
