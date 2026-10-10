from __future__ import annotations

import numpy as np


def test_geometry_identity_and_original_norms(tmp_path, monkeypatch):
    from csasr.experiments import basis_a3_protocol as p
    d = tmp_path / "directions"; d.mkdir()
    for layer in range(32):
        raw = np.array([3.0, 4.0, layer + 1.0])
        cond = np.array([4.0, 3.0, layer + 2.0])
        np.save(d / f"raw_L{layer}.npy", raw)
        np.save(d / f"conditioning_L{layer}.npy", cond)
    monkeypatch.setattr(p, "REPO", tmp_path)
    rows = p.geometry_rows(d)
    assert len(rows) == 32
    for row in rows:
        assert np.isclose(row["unit_l2"], np.sqrt(2 - 2 * row["cos_raw_cond"]))
        assert row["raw_norm"] != 1.0 or row["cond_norm"] != 1.0
