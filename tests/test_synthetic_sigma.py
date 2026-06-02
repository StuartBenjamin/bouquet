"""Golden tests for the synthetic IDA-like fractional-sigma model.

These lock in the deterministic, non-proprietary uncertainty envelope used by
the D3D-like example notebook and the bouquet golden-equilibrium tests.  If the
sine-basis coefficients in ``bouquet.uncertainties._SYNTHETIC_IDA_SIGMA`` are
refit/changed, these golden values must be updated deliberately.
"""
import numpy as np
import pytest

from bouquet import synthetic_ida_sigma

PSI = np.linspace(0.0, 1.0, 201)

# Golden fractional-sigma values [%] at psi_N = 0.0 (core), 0.5 (mid), 1.0 (edge),
# captured from the calibrated fit to DIII-D 204441@4400 IDA.
GOLDEN = {
    "ne": (2.38, 2.38, 5.99),
    "te": (3.17, 5.50, 11.63),
    "ti": (5.69, 7.06, 9.62),
}


@pytest.mark.parametrize("ch,core,mid,edge", [(k, *v) for k, v in GOLDEN.items()])
def test_golden_values(ch, core, mid, edge):
    f = synthetic_ida_sigma(PSI, ch) * 100.0
    assert f[0] == pytest.approx(core, abs=0.05)
    assert f[100] == pytest.approx(mid, abs=0.05)
    assert f[-1] == pytest.approx(edge, abs=0.05)


@pytest.mark.parametrize("ch", ["ne", "te", "ti"])
def test_nonnegative_and_edge_enhanced(ch):
    f = synthetic_ida_sigma(PSI, ch)
    assert np.all(f >= 0.0)                      # physical
    assert f[-1] > f[0]                          # edge enhancement (H-mode pedestal)


def test_ne_is_smooth_no_dip():
    """ne uses no harmonics: flat core, monotone edge rise, never below core."""
    f = synthetic_ida_sigma(PSI, "ne")
    assert f.min() == pytest.approx(f[0], abs=1e-9)             # no sub-core dip/kink
    assert np.all(np.diff(f[PSI >= 0.6]) >= -1e-9)             # monotone toward edge


def test_ni_aliases_ne():
    assert np.allclose(synthetic_ida_sigma(PSI, "ni"),
                       synthetic_ida_sigma(PSI, "ne"))


def test_channel_aliases():
    for alias, canon in [("n_e", "ne"), ("T_e", "te"), ("t_i", "ti"),
                         ("T_12C6", "ti"), ("density", "ne")]:
        assert np.allclose(synthetic_ida_sigma(PSI, alias),
                           synthetic_ida_sigma(PSI, canon))


def test_sol_clamped_to_edge():
    """psi_N > 1 (SOL) holds the LCFS value rather than extrapolating."""
    for ch in ["ne", "te", "ti"]:
        assert synthetic_ida_sigma(np.array([1.3]), ch) == pytest.approx(
            synthetic_ida_sigma(np.array([1.0]), ch))


def test_bad_channel_raises():
    with pytest.raises(ValueError):
        synthetic_ida_sigma(PSI, "bogus")
