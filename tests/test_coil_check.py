"""bouquet.coil_check: windowed measured distribution and candidate likelihood."""

import numpy as np
import pytest

from bouquet.coil_check import CoilCheck, coil_distribution
from bouquet.coil_spec import coil_chi2


def test_window_distribution_and_save(tmp_path):
    t = np.linspace(1.0, 1.1, 201)
    traces = {"F1A": (t, 100.0 + 50.0 * t, np.full(t.size, 7.0)),
              "ECOILA": (t, np.full(t.size, -2e3), None),
              "F2A": (np.array([0.0, 1.05]), np.array([1.0, 2.0]), None)}
    cc = coil_distribution(traces, 1.02, 1.06, turns={"F1A": 58.0}, sigma_extra={"F1A": 3.0})
    w = (t >= 1.02) & (t <= 1.06)
    assert cc.names == ["F1A", "ECOILA"]                     # F2A: one sample in the window
    np.testing.assert_allclose(cc.mean, [58 * np.mean(100 + 50 * t[w]), -2e3])
    np.testing.assert_allclose(
        cc.sigma_meas, [58 * np.sqrt(np.var(50 * t[w], ddof=1) + 49 + 9), 0.0], atol=1e-9)
    cc.save(tmp_path / "cc.npz")
    back = CoilCheck.load(tmp_path / "cc.npz")
    assert back.names == cc.names and back.meta["turns"] == {"F1A": 58.0}
    np.testing.assert_array_equal(back.sigma_meas, cc.sigma_meas)


def test_loglike_matches_coil_chi2():
    cc = CoilCheck(["F1A", "F2A", "F3A", "ECOILA"], [1e4, -2e4, 5e3, 3e4], [50.0, 80.0, 40.0, 500.0])
    names = ["F3A", "ECOILA", "F1A", "F2A"]
    rows = np.array([[5.2e3, 1e9, 1.03e4, -1.98e4],
                     [4.7e3, 3e4, 0.99e4, -2.05e4]])
    model = {"F1A": 300.0, "F2A": 300.0, "F3A": 600.0, "ECOILA": 1e3}
    res = cc.loglike(names, rows, sigma_model=model)
    assert res["coils"] == ["F3A", "F1A", "F2A"]             # ECOILA excluded by default
    sig = {c: np.hypot(s, model[c]) for c, s in zip(cc.names, cc.sigma_meas)}
    for r, row in enumerate(rows):
        ref = coil_chi2(dict(zip(names, row)), dict(zip(cc.names, cc.mean)), sig, coils=res["coils"])
        assert res["chi2"][r] == pytest.approx(ref["chi2"])
        assert res["max_abs_z"][r] == pytest.approx(ref["max_abs_z"])
        assert res["worst_coil"][r] == ref["worst_coil"]
    perm = cc.loglike(names[::-1], rows[:, ::-1], sigma_model=model)
    np.testing.assert_allclose(perm["logL"], res["logL"], rtol=1e-12)
    assert cc.loglike(names, rows, sigma_model=0)["chi2"][0] > res["chi2"][0]
    assert cc.loglike(names, 58 * rows, sigma_model=model)["units_suspect"] == ["F"]
    assert res["units_suspect"] == []
    flat = CoilCheck(["F1A", "F2A"], [1e4, 2e4], [0.0, 50.0])       # e.g. no error, flat trace
    assert flat.loglike(["F1A", "F2A"], [[1.1e4, 2e4]], sigma_model=0)["coils"] == ["F2A"]
