"""Φ_N run: P' scaling through a non-identity map, and the lazy R2 ψ map."""
import numpy as np

from bouquet import coords
from bouquet import TokaMaker_interface as ti


class _Eq:
    """Stand-in solver: psi_bounds plus a toroidal-flux map ψ_N = Φ_N**0.8."""
    psi_bounds = (-0.4, 0.6)

    def __init__(self):
        self.calls = []

    def get_torflux_map(self, x, inverse=False):
        self.calls.append(inverse)
        x = np.asarray(x, float)
        return (x ** 0.8 if inverse else x ** 1.25), np.ones_like(x)


def test_phi_pp_times_dphi_dpsi_is_psi_pp():
    """T3: pp_prof(Φ)·dΦ/dψ ≈ pp_prof(ψ) on the same nodes (PCHIP tolerance)."""
    eq = _Eq()
    x_phi = np.linspace(0.0, 1.0, 401)
    psi = coords.psi_at(eq, x_phi, coords.PHI)
    p = (1.0 - psi) ** 2 * (1.0 + psi)            # p(ψ) sampled on the Φ nodes
    pp_phi = coords.pp_prof(eq, x_phi, p, coords.PHI)["y"]
    pp_psi = coords.pp_prof(eq, psi, p)["y"]
    dphi_dpsi = 1.25 * psi ** 0.25                 # Φ = ψ**1.25
    inner = slice(5, -2)   # dψ/dΦ singular at the axis; one-sided PCHIP ends
    np.testing.assert_allclose(pp_phi[inner] * dphi_dpsi[inner], pp_psi[inner],
                               rtol=5e-3, atol=2e-3)
    # analytic dp/dψ / psi_range
    exact = (-2 * (1 - psi) * (1 + psi) + (1 - psi) ** 2) / 1.0
    np.testing.assert_allclose(pp_psi[inner], exact[inner], rtol=5e-3, atol=2e-3)


class _Anchor:
    def solve_scale(self, j_ind, j_other):
        return 1.5


def test_r2_ip_scale_maps_psi_only_on_the_legacy_branch(monkeypatch):
    """S2: the anchor path never evaluates the ψ map; legacy maps Φ → ψ."""
    eq = _Eq()
    x = np.linspace(0.0, 1.0, 11)
    j = np.ones_like(x)
    assert ti._r2_ip_scale(_Anchor(), eq, j, j, x, 1.0, coords.PHI) == 1.5
    assert eq.calls == []

    seen = {}

    def _f(alpha, mygs, j_ind, j_other, psi_N, Ip_target):
        seen["psi"] = psi_N
        return alpha - 2.0 * Ip_target

    monkeypatch.setattr(ti, "Ip_flux_integral_vs_target", _f)
    s = ti._r2_ip_scale(None, eq, j, j, x, 1.0, coords.PHI)
    assert abs(s - 2.0) < 1e-5
    np.testing.assert_allclose(seen["psi"], x ** 0.8)
    ti._r2_ip_scale(None, eq, j, j, x, 1.0)
    assert seen["psi"] is x                        # psi_n: the grid itself

