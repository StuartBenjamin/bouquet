"""Unit tests for bouquet.physics -- pure numpy, no OFT/solver required.

Covers the two convention reductions used by the baseline resolvers and the
per-draw bootstrap recompute:

  * isotropize_fast_pressure -- anisotropic fast pressure -> scalar GS pressure
  * parallel_to_toroidal     -- FSA parallel current <j.B> -> toroidal
                                <j_phi/R>/<1/R>, ratio + analytic methods
"""

import numpy as np
import pytest

from bouquet.physics import isotropize_fast_pressure, parallel_to_toroidal


# ---------------------------------------------------------------------------
# isotropize_fast_pressure
# ---------------------------------------------------------------------------
class TestIsotropizeFastPressure:
    def setup_method(self):
        self.p_perp = np.array([3.0e4, 2.0e4, 0.0])
        self.p_par = np.array([1.5e4, 2.0e4, 0.0])

    def test_trace(self):
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="trace")
        assert np.allclose(out, (2 * self.p_perp + self.p_par) / 3.0)

    def test_mean(self):
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="mean")
        assert np.allclose(out, (self.p_perp + self.p_par) / 2.0)

    def test_perp(self):
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="perp")
        assert np.allclose(out, self.p_perp)

    def test_isotropic_input_is_identity(self):
        # p_perp == p_par -> every reduction returns the same scalar pressure
        p = np.array([1.0e4, 5.0e3])
        for method in ("trace", "mean", "perp"):
            assert np.allclose(isotropize_fast_pressure(p, p, method=method), p)

    def test_trace_preserves_energy_density(self):
        # w = (p_par + 2 p_perp)/2 = (3/2) p_scalar for the trace reduction
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="trace")
        w = 0.5 * (self.p_par + 2 * self.p_perp)
        assert np.allclose(1.5 * out, w)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            isotropize_fast_pressure(np.zeros(3), np.zeros(4))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown p_fast reduction"):
            isotropize_fast_pressure(self.p_perp, self.p_par, method="rms")


# ---------------------------------------------------------------------------
# parallel_to_toroidal -- ratio method
# ---------------------------------------------------------------------------
class TestParallelToToroidalRatio:
    def test_component_scales_with_total_ratio(self):
        j_par_tot = np.array([2.0e6, 1.0e6, 5.0e5])
        j_tor_tot = np.array([1.8e6, 0.95e6, 4.6e5])
        j_comp = np.array([1.0e5, 2.0e5, 3.0e5])
        out = parallel_to_toroidal(
            j_comp, j_parallel_total=j_par_tot, j_tor_total=j_tor_tot)
        assert np.allclose(out, j_comp * j_tor_tot / j_par_tot)

    def test_zero_crossing_filled_from_neighbours(self):
        # surface where the total parallel current passes through zero:
        # the ill-defined ratio must be interpolated, not propagated as inf
        j_par_tot = np.array([2.0e6, 0.0, 1.0e6])
        j_tor_tot = np.array([1.0e6, 0.0, 0.5e6])
        j_comp = np.ones(3)
        out = parallel_to_toroidal(
            j_comp, j_parallel_total=j_par_tot, j_tor_total=j_tor_tot)
        assert np.all(np.isfinite(out))
        assert np.isclose(out[1], 0.5)  # interpolated between c=0.5 and c=0.5

    def test_all_zero_total_raises(self):
        with pytest.raises(ValueError, match="cannot form ratio"):
            parallel_to_toroidal(
                np.ones(3), j_parallel_total=np.zeros(3), j_tor_total=np.zeros(3))


# ---------------------------------------------------------------------------
# parallel_to_toroidal -- analytic (field-aligned) method
# ---------------------------------------------------------------------------
class TestParallelToToroidalAnalytic:
    def test_cylinder_limit_is_identity(self):
        # B_p -> 0 and no R variation: j_tor == j_parallel exactly
        R0, B0 = 1.7, 2.0
        F = R0 * B0
        j_par = np.array([1.0e6, 5.0e5, 2.0e5])
        geom = {
            "F": np.full(3, F),
            "avg_inv_R": np.full(3, 1.0 / R0),
            "avg_B2": np.full(3, B0**2),
            "avg_inv_R2": np.full(3, 1.0 / R0**2),
        }
        out = parallel_to_toroidal(j_par * B0, geom=geom)  # raw <j.B>
        assert np.allclose(out, j_par)

    def test_imas_b0_normalisation(self):
        # IMAS convention input <j.B>/B0 with geom["B0"] must equal raw <j.B>
        R0, B0 = 1.7, 2.0
        geom = {
            "F": R0 * B0, "avg_inv_R": 1.0 / R0,
            "avg_B2": B0**2, "avg_inv_R2": 1.0 / R0**2,
        }
        j_par_imas = np.array([1.0e6])
        raw = parallel_to_toroidal(j_par_imas * B0, geom=dict(geom))
        viaB0 = parallel_to_toroidal(j_par_imas, geom={**geom, "B0": B0})
        assert np.allclose(raw, viaB0)

    def test_exact_formula_on_shaped_surface(self):
        # j_tor = <j.B> F <1/R^2> / (<B^2> <1/R>) reproduced exactly when
        # <1/R^2> is supplied
        R0, eps, F = 1.7, 0.36, 3.4
        th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        Rs = R0 * (1 + eps * np.cos(th))
        inv_R, inv_R2 = np.mean(1 / Rs), np.mean(1 / Rs**2)
        B2 = F**2 * inv_R2 * 1.008  # ~0.8% poloidal-field content
        jB = 1.0e6
        out = parallel_to_toroidal(
            np.array([jB]),
            geom={"F": F, "avg_inv_R": inv_R, "avg_B2": B2, "avg_inv_R2": inv_R2},
        )
        assert np.isclose(out[0], jB * F * inv_R2 / (B2 * inv_R), rtol=1e-12)

    def test_missing_inv_R2_error_is_order_Bp_over_B_squared(self):
        # dropping <1/R^2> must only cost the <B_p^2>/<B^2> bracket (<~1%)
        R0, eps, F = 1.7, 0.36, 3.4
        th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        Rs = R0 * (1 + eps * np.cos(th))
        inv_R, inv_R2 = np.mean(1 / Rs), np.mean(1 / Rs**2)
        bp_frac = 0.008  # <B_p^2>/<B_phi^2>
        B2 = F**2 * inv_R2 * (1 + bp_frac)
        jB = np.array([1.0e6])
        geom = {"F": F, "avg_inv_R": inv_R, "avg_B2": B2}
        exact = parallel_to_toroidal(jB, geom={**geom, "avg_inv_R2": inv_R2})
        approx = parallel_to_toroidal(jB, geom=geom)
        rel = abs(approx[0] - exact[0]) / abs(exact[0])
        assert rel < 0.01
        assert np.isclose(rel, bp_frac, rtol=0.05)

    def test_cocos_sign_safety(self):
        # flipping the signs of F and <j.B> together (opposite-helicity COCOS)
        # must not flip or distort the toroidal output magnitude
        R0, F = 1.7, 3.4
        geom = {"F": F, "avg_inv_R": 1.0 / R0, "avg_B2": (F / R0) ** 2}
        out_pos = parallel_to_toroidal(np.array([1.0e6]), geom=dict(geom))
        out_neg = parallel_to_toroidal(np.array([-1.0e6]), geom={**geom, "F": -F})
        assert np.isclose(out_pos[0], out_neg[0])

    def test_missing_geom_key_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            parallel_to_toroidal(np.ones(2), geom={"F": 1.0})

    def test_no_method_selected_raises(self):
        with pytest.raises(ValueError, match="ratio method"):
            parallel_to_toroidal(np.ones(2))


# ---------------------------------------------------------------------------
# impurity derivation -- effective_impurity_charge / main_ion_density_from_zeff
# ---------------------------------------------------------------------------
from bouquet.physics import effective_impurity_charge, main_ion_density_from_zeff


class TestImpurityDerivation:
    def test_round_trip_consistent_baseline(self):
        # build a consistent (ne, ni, zeff) set from known carbon Z=6, then
        # recover Z_imp and re-derive ni exactly
        Z = 6.0
        psi = np.linspace(0, 1, 50)
        ne = 5e19 * (1 - 0.8 * psi**2)
        zeff = 2.0 + 0.3 * psi          # 2.0 .. 2.3
        ni = ne * (Z - zeff) / (Z - 1.0)
        Z_rec = effective_impurity_charge(ne, ni, zeff)
        assert Z_rec is not None
        assert np.isclose(Z_rec, Z, rtol=1e-10)
        ni_rec = main_ion_density_from_zeff(ne, zeff, Z_rec)
        assert np.allclose(ni_rec, ni)

    def test_no_dilution_returns_none(self):
        # the IDA ni = ne workflow carries no dilution information
        ne = np.full(20, 4e19)
        assert effective_impurity_charge(ne, ne.copy(), np.full(20, 1.8)) is None

    def test_physical_bounds(self):
        # for 1 <= zeff <= Z_imp: 0 <= ni <= ne and nz >= 0
        Z = 6.0
        ne = np.full(11, 5e19)
        zeff = np.linspace(1.0, Z, 11)
        ni = main_ion_density_from_zeff(ne, zeff, Z)
        nz = (ne - ni) / Z
        assert np.all(ni >= -1e-6) and np.all(ni <= ne + 1e-6)
        assert np.all(nz >= -1e-6)
        assert np.isclose(ni[0], ne[0])     # zeff=1 -> pure plasma
        assert np.isclose(ni[-1], 0.0)      # zeff=Z -> fully diluted

    def test_invalid_z_imp_raises(self):
        with pytest.raises(ValueError, match="exceed 1"):
            main_ion_density_from_zeff(np.ones(3), np.ones(3), 1.0)

    def test_zeff_floor_ignored_in_median(self):
        # surfaces with zeff <= 1 or negligible dilution must not poison Z_imp
        Z = 6.0
        ne = np.full(30, 5e19)
        zeff = np.full(30, 2.0)
        ni = ne * (Z - zeff) / (Z - 1.0)
        zeff_noisy = zeff.copy(); zeff_noisy[:3] = 0.99   # bad edge points
        ni_noisy = ni.copy(); ni_noisy[3:6] = ne[3:6]     # zero-dilution points
        Z_rec = effective_impurity_charge(ne, ni_noisy, zeff_noisy)
        assert Z_rec is not None and np.isclose(Z_rec, Z, rtol=1e-9)
