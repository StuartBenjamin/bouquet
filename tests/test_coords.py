"""bouquet.coords: the run coordinate at the OpenFUSIONToolkit boundary."""
import numpy as np
import pytest

from bouquet import coords


class _Eq:
    """Stand-in solver: psi_bounds plus a toroidal-flux map ψ_N = Φ_N**0.8."""
    psi_bounds = (-0.4, 0.6)

    def __init__(self):
        self.calls = []

    def get_torflux_map(self, x, inverse=False):
        self.calls.append(inverse)
        x = np.asarray(x, float)
        return (x ** 0.8 if inverse else x ** 1.25), np.ones_like(x)


@pytest.fixture
def swb_params(monkeypatch):
    def _set(names):
        monkeypatch.setattr(coords, "_SWB_PARAMS", frozenset(names), raising=False)
    return _set


X = np.array([0.0, 0.05, 0.2, 0.5, 0.9, 1.0])


def test_psi_profile_dicts_carry_no_coord():
    d = coords.oft_prof("jphi-linterp", X, X ** 2)
    assert set(d) == {"type", "x", "y"}
    pp = coords.pp_prof(_Eq(), X, 1.0 - X ** 2)
    assert "coord" not in pp and pp["type"] == "linterp"


def test_phi_profile_tags():
    assert coords.oft_prof("jphi-linterp", X, X, coords.PHI)["coord"] == "phi_n_relabel"
    assert coords.oft_prof("linterp", X, X, coords.PHI)["coord"] == "phi_n"


def test_pp_scale_is_the_same_in_both_coordinates():
    p = 1.0 - X ** 2
    a = coords.pp_prof(_Eq(), X, p)["y"]
    b = coords.pp_prof(_Eq(), X, p, coords.PHI)["y"]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[1:-1], -2 * X[1:-1] / 1.0, rtol=0.05)


def test_psi_of():
    eq = _Eq()
    np.testing.assert_array_equal(coords.psi_of(eq, X, psi_pad=0.01),
                                  np.clip(X, 0.01, 0.99))
    assert eq.calls == []
    got = coords.psi_of(eq, X, coords.PHI, psi_pad=0.01)
    np.testing.assert_allclose(got, np.clip(X ** 0.8, 0.01, 0.99))
    assert eq.calls == [True]


def test_window_x():
    eq = _Eq()
    for wc in ("psi_n", "native"):
        np.testing.assert_array_equal(coords.window_x(eq, X, coords.PSI, wc), X)
    np.testing.assert_array_equal(coords.window_x(eq, X, coords.PHI, "native"), X)
    np.testing.assert_allclose(coords.window_x(eq, X, coords.PHI, "psi_n"), X ** 0.8)
    with pytest.raises(ValueError):
        coords.window_x(eq, X, coords.PSI, "rho")


def test_rho_tor_input_becomes_phi_n():
    c, x = coords.resolve_input_coord("rho_tor", np.array([0.0, 0.5, 1.0]))
    assert c == coords.PHI
    np.testing.assert_array_equal(x, [0.0, 0.25, 1.0])
    assert coords.resolve_input_coord("psi_n", X)[1] is X
    with pytest.raises(ValueError):
        coords.resolve_input_coord("psi", X)


def test_swb_grid_on_a_toolkit_with_psi_N(swb_params):
    swb_params({"mygs", "ne", "psi_N"})
    xi = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_array_equal(coords.swb_grid(xi), xi)
    assert list(coords.swb_grid_kwargs(xi)) == ["psi_N"]
    assert coords.swb_grid_kwargs(xi, coords.PHI)["coord"] == coords.PHI
    np.testing.assert_array_equal(coords.swb_seed(xi), (1 - xi ** 1.5) ** 1.5)


def test_swb_grid_on_a_legacy_toolkit(swb_params):
    swb_params({"mygs", "ne"})
    xi = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_array_equal(coords.swb_grid(xi), np.linspace(0, 1, 4))
    assert coords.swb_grid_kwargs(xi) == {}


def test_seed_matches_oft_power_flux_fun_on_a_uniform_grid(swb_params):
    swb_params({"psi_N"})
    s = np.linspace(0.0, 1.0, 129)
    ref = np.power(1.0 - np.power(np.linspace(0.0, 1.0, 129), 1.5), 1.5)
    np.testing.assert_array_equal(coords.swb_seed(s), ref)


def test_check_backend():
    coords.check_backend(coords.PSI)
    with pytest.raises(ValueError):
        coords.check_backend("rho_tor")
