"""ida_hybrid: the run-grid ni is main_ion_density_from_zeff of the run-grid
(ne, Z_eff), so the baseline stays quasineutral between IDA's nodes.

Synthetic inputs only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import bouquet.io.imas as I
from bouquet.physics import main_ion_density_from_zeff

_Z = 6.0


def _ida():
    x = np.linspace(0.0, 1.0, 9)                      # coarse IDA grid
    ne = 5e19 * (1.0 - 0.6 * x ** 2) + 1e18
    zeff = 1.6 + 0.8 * x ** 3
    ni = main_ion_density_from_zeff(ne, zeff, _Z)
    s = 0.05 * ne
    return SimpleNamespace(psi_N=x, ne=ne, te=1e3 * (1 - x) + 50, ti=1e3 * (1 - x) + 50,
                           ni=ni, Zeff=zeff, sigma_ne=s, sigma_te=s, sigma_ni=s,
                           sigma_ti=s, q=None)


def _merge(monkeypatch, ida, **kw):
    import bouquet.io.ida as IDA
    monkeypatch.setattr(IDA, "read_ida", lambda *a, **k: ida)
    monkeypatch.setattr(I, "_read_ida_omega", lambda *a, **k: None)
    psi = np.linspace(0.0, 1.0, 101)                  # finer run grid
    # the EXPERIMENTAL IDA route (PR #56): ni rebuilt from IDA's (ne, Z_eff)
    return psi, I._merge_ida_kinetics(psi, None, None, None, "x.cdf", 1.0, _Z,
                                      ni_source="all", **kw)


def test_the_run_grid_ni_is_the_formula_of_the_run_grid_ne_and_zeff(monkeypatch):
    psi, out = _merge(monkeypatch, _ida())
    ne, ni, zeff = out[0], out[3], out[4]
    np.testing.assert_allclose(ni, main_ion_density_from_zeff(ne, zeff, _Z),
                               rtol=1e-14, atol=0.0)
    # ... and on IDA's own nodes it is IDA's ni
    xi = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(ni[np.isin(psi, xi)],
                               _ida().ni[np.isin(xi, psi)], rtol=1e-13)


def test_the_ida_profiles_are_regridded_c1(monkeypatch):
    """IDA's kinetics reach the run grid through pchip_interp (C1): a linear
    regrid kinks every profile at each IDA knot, and P' (FF') then jumps there."""
    from bouquet.utils import pchip_interp
    ida = _ida()
    psi, out = _merge(monkeypatch, ida)
    for got, src in zip(out[:3], (ida.ne, ida.te, ida.ti)):
        np.testing.assert_allclose(got, pchip_interp(ida.psi_N, src, psi), rtol=1e-14)
    slope = np.diff(out[0]) / np.diff(psi)
    knot = np.searchsorted(psi, 0.5)                  # an IDA knot that is a run node
    assert abs(slope[knot] - slope[knot - 1]) < 0.05 * abs(slope[knot])


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_the_ida_regrid_is_the_configured_method(monkeypatch, method):
    """ImasSource.ida_regrid picks how IDA's kinetics and their sigma reach
    the run grid (ledger B-24: pchip)."""
    from bouquet.utils import regrid
    ida = _ida()
    psi, out = _merge(monkeypatch, ida, ida_regrid=method)
    for got, src in zip(out[:3] + out[6:10], (ida.ne, ida.te, ida.ti, ida.sigma_ne,
                                              ida.sigma_te, ida.sigma_ni, ida.sigma_ti)):
        np.testing.assert_allclose(got, regrid(ida.psi_N, src, psi, method), rtol=1e-14)


def test_an_unknown_ida_regrid_is_refused():
    from bouquet.config import ImasSource
    with pytest.raises(ValueError, match="ida_regrid"):
        ImasSource(ids_path="dd.json", time=1.0, ida_path="x.cdf", ida_regrid="akima")
