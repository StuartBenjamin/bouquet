"""The boundary constraint's calibrated balance (bouquet.boundary_balance) --
OFT-free half.

The live probes (the partition probe and the draw-yield probe on the
synthetic fixtures) are tests/test_boundary_balance_solver.py.
"""
import math
import os

import numpy as np
import pytest

from bouquet import boundary_balance as BB
from bouquet import solver_geometry as SG

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
#  the analytic rule
# ---------------------------------------------------------------------------
def test_the_analytic_total_is_the_squared_penalty_balance():
    # sum_j (W u_j rbp_j delta)^2 == sum_c (r_c mu0 dI_c)^2 at W = W_a
    n = 64
    u = np.full(n, 1.0 / n)
    rbp = np.linspace(0.2, 0.6, n)
    cuts = {"A": 1500.0, "B": 900.0, "C": 300.0}
    reg = {"A": 1e4, "B": 1e4, "C": 2e3}
    W, parts = BB.analytic_weight_total(u, rbp, 8.5e-3, cuts, reg)
    lhs = np.sum((W * u[1:] * rbp[1:] * 8.5e-3) ** 2)
    rhs = sum((reg[c] * BB.MU0 * cuts[c]) ** 2 for c in cuts)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert parts["n_points_applied"] == n - 1 and parts["n_coils"] == 3


def test_the_reference_point_is_not_in_the_boundary_sum():
    u = np.array([0.5, 0.25, 0.25])
    a, _ = BB.analytic_weight_total(u, [1e6, 1.0, 1.0], 1e-3, {"A": 1.0},
                                    {"A": 1.0})
    b, _ = BB.analytic_weight_total(u, [0.0, 1.0, 1.0], 1e-3, {"A": 1.0},
                                    {"A": 1.0})
    assert a == b


@pytest.mark.parametrize("k", [2.0, 10.0])
def test_the_analytic_total_scales_as_the_rule_says(k):
    u = np.full(16, 1 / 16)
    rbp = np.full(16, 0.4)
    base, _ = BB.analytic_weight_total(u, rbp, 5e-3, {"A": 1e3}, {"A": 1e4})
    assert BB.analytic_weight_total(u, rbp, k * 5e-3, {"A": 1e3},
                                    {"A": 1e4})[0] == pytest.approx(base / k)
    assert BB.analytic_weight_total(u, rbp, 5e-3, {"A": k * 1e3},
                                    {"A": 1e4})[0] == pytest.approx(base * k)
    assert BB.analytic_weight_total(u, k * rbp, 5e-3, {"A": 1e3},
                                    {"A": 1e4})[0] == pytest.approx(base / k)


def test_no_regularised_coil_or_no_field_gives_no_estimate():
    u = np.full(4, 0.25)
    assert math.isnan(BB.analytic_weight_total(u, np.ones(4), 1e-3,
                                               {"A": 1.0}, {})[0])
    assert math.isnan(BB.analytic_weight_total(u, np.full(4, np.nan), 1e-3,
                                               {"A": 1.0}, {"A": 1.0})[0])


def test_the_coil_cut_and_fraction_are_one_exactly_at_the_cut():
    base = {"A": 1e4, "F9A": -2e3}
    sig = {"A": 100.0, "F9A": 50.0}
    cuts = BB.coil_cuts_for("chi2", base, sig, chi2_max=6.1)
    assert cuts["A"] == pytest.approx(math.sqrt(6.1) * 100.0)
    # every coil exactly at its cut: chi2/nu = chi2_max
    z = {c: cuts[c] / sig[c] for c in sig}
    chi2_nu = float(np.mean(np.square(list(z.values()))))
    assert BB.coil_fraction("chi2", dict(chi2_nu=chi2_nu, max_abs_z=0.0),
                            chi2_max=6.1, z_max=6.3) == pytest.approx(1.0)
    assert BB.coil_fraction("chi2", dict(chi2_nu=0.0, max_abs_z=6.3),
                            chi2_max=6.1, z_max=6.3) == pytest.approx(1.0)
    lg = BB.coil_cuts_for("legacy", base, inspec_F_max=0.02,
                          inspec_VSC_max=0.01, vsc_coils=("F9A",))
    assert lg == {"A": pytest.approx(200.0), "F9A": pytest.approx(20.0)}
    d = BB.drift_pcts({"A": 1.02e4, "F9A": -2.02e3}, base, ("F9A",))
    assert BB.coil_fraction("legacy", d, inspec_F_max=0.02,
                            inspec_VSC_max=0.01) == pytest.approx(1.0)
    assert math.isnan(BB.coil_fraction("chi2", {}, chi2_max=6.1))


# ---------------------------------------------------------------------------
#  the root find on log W
# ---------------------------------------------------------------------------
def _model(Wstar, alpha=1.0, fail=lambda W: False):
    calls = []

    def ev(W):
        calls.append(W)
        if fail(W):
            raise RuntimeError("engine GS solve failed: Exceeded maxits")
        return dict(rho=(Wstar / W) ** alpha)
    return ev, calls


@pytest.mark.parametrize("Wstar", [2e3, 4e4, 3e5, 5e6])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_the_root_find_brings_rho_within_tolerance(Wstar, alpha):
    ev, calls = _model(Wstar, alpha)
    r = BB.calibrate_log_root(ev, 1e5)
    assert r["status"] == "converged"
    assert 1 / BB.BALANCE_TOLERANCE <= r["rho"] <= BB.BALANCE_TOLERANCE
    assert r["n_solves"] == len(calls) <= BB.PROBE_MAX_SOLVES
    assert r["W"] in calls                      # a MEASURED point, never extrapolated


def test_a_start_within_tolerance_costs_one_solve():
    ev, calls = _model(1.1e5)
    r = BB.calibrate_log_root(ev, 1e5)
    assert r["status"] == "converged" and r["W"] == 1e5 and len(calls) == 1


def test_a_root_beyond_the_reach_is_no_bracket():
    ev, _ = _model(1e10)
    r = BB.calibrate_log_root(ev, 1e5)
    assert r["status"] == "no_bracket" and r["W"] is None


def test_failed_points_shorten_the_step_and_are_recorded():
    # the stiff side fails (the I4 symptom: a too-strong boundary does not
    # converge); the root lies between
    ev, calls = _model(3e5, fail=lambda W: W > 2e6)
    r = BB.calibrate_log_root(ev, 1e5)
    assert any(p.get("error") for p in r["points"]) or r["status"] == "converged"
    assert r["status"] == "converged"
    ev2, _ = _model(3e5, fail=lambda W: True)
    r2 = BB.calibrate_log_root(ev2, 1e5)
    assert r2["status"] == "no_bracket" and r2["W"] is None
    assert all("Exceeded maxits" in p["error"] for p in r2["points"])


def test_a_budget_ends_on_the_best_measured_point():
    calls = []

    def ev(W):                       # not a power law: regula falsi needs steps
        calls.append(W)
        return dict(rho=(3.3e5 / W) ** (2.0 if W < 3.3e5 else 0.5))
    r = BB.calibrate_log_root(ev, 1e5, tol=1.0001, max_solves=3)
    assert r["status"] == "budget" and r["W"] in calls
    best = min(calls, key=lambda W: abs(math.log(3.3e5 / W)))
    assert r["W"] == best


# ---------------------------------------------------------------------------
#  the geometry: provisional, override, calibrated -- and the stamps
# ---------------------------------------------------------------------------
def _square(n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([1.7 + 0.6 * np.cos(t), 1.2 * np.sin(t)])


class _Solver:
    F0 = None
    isoflux_pts = None
    isoflux_weights = None
    saddle_targets = None
    saddle_weights = None
    isoflux_weight_total = None


def _geom(monkeypatch, override=None):
    from bouquet.config import ReconstructionSource

    class _G:
        R_center, B_center = 1.7, 2.0
        boundary_R, boundary_Z = _square().T
    import bouquet.io.geqdsk as gq
    monkeypatch.setattr(gq, "read_geqdsk", lambda *a, **k: _G)
    src = ReconstructionSource.__new__(ReconstructionSource)
    src.geqdsk_path, src.cocos = "x", None
    s = _Solver()
    s.isoflux_weight_total = override
    return SG.case_geometry(src, s, standard=True)


def test_the_default_geometry_is_provisional_and_calibration_is_on(monkeypatch):
    from bouquet.engine import balance_calibration_on
    g = _geom(monkeypatch)
    rec = g.record()
    assert rec["isoflux_weight_mode"] == "provisional"
    assert rec["isoflux_weight_total"] == pytest.approx(SG.ISOFLUX_WEIGHT_TOTAL)
    assert balance_calibration_on(g)


def test_an_override_is_installed_as_given_and_never_calibrated(monkeypatch):
    from bouquet.engine import balance_calibration_on
    g = _geom(monkeypatch, override=5e4)
    rec = g.record()
    assert rec["isoflux_weight_mode"] == "override"
    assert rec["isoflux_weight_total"] == pytest.approx(5e4)
    assert rec["isoflux_balance"]["source"] == "SolverConfig.isoflux_weight_total"
    assert not balance_calibration_on(g)


def test_the_calibrated_set_keeps_points_and_shape_and_stamps_both_totals(
        monkeypatch):
    g = _geom(monkeypatch)
    brec = dict(mode="calibrated", total=7.5e4, analytic_total=1e5,
                partition_ratio=0.97, status="converged")
    c = SG.with_balanced_total(g, 7.5e4, brec)
    assert np.array_equal(c.iso_pts, g.iso_pts)
    np.testing.assert_allclose(c.iso_w / c.iso_w.sum(),
                               g.iso_w / g.iso_w.sum(), rtol=1e-14)
    rec = c.record()
    assert rec["isoflux_weight_mode"] == "calibrated"
    assert rec["isoflux_weight_total"] == pytest.approx(7.5e4)
    # F5: the reference point's weight is not applied by TokaMaker
    assert rec["isoflux_weight_installed_total"] == pytest.approx(
        7.5e4 - c.iso_w[0])
    assert rec["isoflux_balance"]["partition_ratio"] == 0.97
    assert rec["isoflux_resampling"]["weight_total"] == pytest.approx(7.5e4)
    assert rec["isoflux_sha256"] != g.record()["isoflux_sha256"]
    from bouquet.engine import balance_calibration_on
    assert not balance_calibration_on(c)       # never re-calibrated in place


def test_the_override_field_exists_and_defaults_to_calibrate():
    from bouquet.config import SolverConfig
    assert SolverConfig(mesh_path="m").isoflux_weight_total is None
