"""The boundary constraint's strength (a constant inside a measured window),
its report-only partition probe (bouquet.boundary_balance), the homotopy's
#VSC bound floor and the vertical-runaway label -- OFT-free half.

The live probes (the stamped partition and the draw yield on the synthetic
fixtures) are tests/test_boundary_balance_solver.py.
"""
import inspect
import math
import os
import types

import numpy as np
import pytest

from bouquet import boundary_balance as BB
from bouquet import solver_geometry as SG

_HERE = os.path.dirname(os.path.abspath(__file__))


def test_the_legacy_partition_fraction_is_the_inspec_legacy_metric():
    """Review of #94, C3: same coil set (F* only, the pair out) and the same
    1 A floor as the in-spec legacy F metric; a small non-F coil's change,
    which the metric never judges, leaves the fraction alone."""
    from bouquet.TokaMaker_interface import _coil_drift_pct, _coil_max_drifts
    base = {"F1A": 5.0e4, "F2A": -3.0e4, "ECOILA": 0.25, "F9A": -6.6e4,
            "F9B": 2.8e3, "F5B": 0.4}
    cur = {"F1A": 5.05e4, "F2A": -3.0e4, "ECOILA": 0.5, "F9A": -6.0e4,
           "F9B": 2.0e3, "F5B": 0.9}
    vsc = ("F9A", "F9B")
    metric, _ = _coil_max_drifts(_coil_drift_pct(cur, base), vsc)
    got = BB.drift_pcts(cur, base, vsc)["max_F_drift_pct"]
    assert got == metric
    # the sub-amp F coil is judged on the 1 A floor (50 %), not on |I0|
    # (125 %); the 0.25 A ECOILA (100 % of itself) is not judged at all
    assert got == pytest.approx(50.0)
    # a coil set with no F* coil: the metric's 0, not a number from other coils
    assert BB.drift_pcts({"E1": 2.0, "F9A": 0.0}, {"E1": 1.0, "F9A": 1.0},
                         vsc)["max_F_drift_pct"] == 0.0


# ---------------------------------------------------------------------------
#  the constants and their recorded measurements
# ---------------------------------------------------------------------------
def test_the_boundary_total_is_the_constant_with_its_measured_window():
    # 2e5 is a constant inside a MEASURED window (owner decision 2026-10-16);
    # the window and the balance band are part of its definition: a change
    # of the constant must come with a change of this record
    assert SG.ISOFLUX_WEIGHT_TOTAL == 200_000
    src = inspect.getsource(SG)
    block = src[src.index("#: The TOTAL isoflux weight"):
                src.index("ISOFLUX_WEIGHT_TOTAL = ")]
    for text in ("MEASURED WINDOW", "1.5e5-4e5", "1.8e5-3.8e5",
                 "balance-crossing band"):
        assert text in block, text


# ---------------------------------------------------------------------------
#  the report-only partition
# ---------------------------------------------------------------------------
def test_the_coil_fraction_is_one_at_the_cut_and_leaves_the_vsc_pair_out():
    sig = {"A": 100.0, "B": 80.0, "F9A": 50.0, "F9B": 50.0}
    base = {"A": 1e4, "B": -5e3, "F9A": -6e4, "F9B": 3e3}
    cuts = dict(kind="chi2", sigma=sig, vsc_coils=("F9A", "F9B"),
                delta_cut_mm=5.0,
                frac_kw=dict(chi2_max=6.1, z_max=6.3, inspec_F_max=0.02,
                             inspec_VSC_max=0.01))
    s = math.sqrt(6.1)
    cur = dict(A=1e4 + s * 100.0, B=-5e3 + s * 80.0, F9A=-6e4 + 5e3,
               F9B=3e3 - 5e3)            # the pair far off: not counted
    rho, bf, cf, info = BB.partition_of(cur, base, 5.0, cuts)
    assert cf == pytest.approx(1.0) and bf == pytest.approx(1.0)
    assert rho == pytest.approx(1.0) and info["nu"] == 2
    # the legacy fraction is the legacy in-spec F metric, which judges coils
    # named F* only (review of #94, C3): the same numbers on F-named coils
    lg = dict(cuts, kind="legacy")
    lbase = {"F1A": 1e4, "F2B": -5e3, "F9A": -6e4, "F9B": 3e3}
    lcur = {"F1A": 1.02e4, "F2B": -5e3, "F9A": 0.0, "F9B": 0.0}
    d = BB.drift_pcts(lcur, lbase, ("F9A", "F9B"))
    assert BB.coil_fraction("legacy", d, **lg["frac_kw"]) == \
        pytest.approx(1.0)
    rho2, _, _, _ = BB.partition_of(lcur, lbase, 10.0, lg)
    assert rho2 == pytest.approx(2.0)
    assert math.isnan(BB.coil_fraction("chi2", {}, chi2_max=6.1))


@pytest.mark.parametrize("rho, ok", [(1.0, True), (0.34, True), (2.9, True),
                                     (0.3, False), (3.2, False),
                                     (float("nan"), None), (None, None)])
def test_the_partition_band(rho, ok):
    assert BB.PARTITION_BAND == (pytest.approx(1 / 3), 3.0)
    assert BB.in_band(rho) is ok


def _fake_case(monkeypatch, solve_raises=False, coil_shift=0.0, rms=2.0):
    import bouquet.filtering as F
    import bouquet.utils as U
    calls = []
    # F-named coils: the legacy fraction is the legacy in-spec F metric,
    # which judges coils named F* only (review of #94, C3)
    base = {"F1A": 1e4, "F2B": -5e3, "F9A": -6e4, "F9B": 3e3}
    state = dict(cur=dict(base))

    class _GS:
        coil_sets = list(base)

        def copy_eq(self):
            return "DELIVERED"

        def replace_eq(self, source_eq=None):
            calls.append(("replace_eq", source_eq))
            state["cur"] = dict(base)

        def trace_surf(self, psi):
            return None

        def get_coil_currents(self):
            return dict(state["cur"]), None

        def set_coil_reg(self, reg_terms=None):
            calls.append(("set_coil_reg", len(reg_terms)))

        def coil_reg_term(self, coffs, target=0.0, weight=1.0):
            return (coffs, target, weight)

    class _Backend:
        def set_inputs(self, pressure=None, kinetics=None):
            calls.append(("pressure", float(np.max(pressure))))

        def solve(self, request, n_passes=1):
            calls.append(("solve", n_passes))
            if solve_raises:
                raise RuntimeError("engine GS solve failed: maxits")
            state["cur"] = dict(base, F1A=base["F1A"] * (1 + coil_shift))

    monkeypatch.setattr(U, "safe_trace_surf", lambda gs, psi: "LCFS")
    monkeypatch.setattr(F, "boundary_deviation_mm", lambda a, b: (rms, 2 * rms))
    fc = types.SimpleNamespace(coil_filter="legacy", inspec_F_max=0.02,
                               inspec_VSC_max=0.01)
    bq = types.SimpleNamespace(
        mygs=_GS(), config=types.SimpleNamespace(
            filtering=fc, solver=types.SimpleNamespace(
                coil_vsc={"F9A": 1.0, "F9B": -1.0})),
        _boundary_cut=lambda quiet=True: (5.0, "test"))
    return bq, _Backend, calls


def test_the_probe_measures_once_restores_and_never_changes_the_total(
        monkeypatch):
    bq, Backend, calls = _fake_case(monkeypatch, coil_shift=0.02, rms=5.0)
    rec = BB.measure_partition(
        bq, np.ones(3), np.full(3, 1e4), lambda maxits: Backend(),
        psi_pad=1e-3, maxits=100, coil_reg_terms=["RECON"],
        geom_record=dict(isoflux_weight_total=2e5,
                         isoflux_weight_installed_total=199_219.0))
    assert rec["status"] == "measured" and rec["n_solves"] == 1
    assert rec["partition"] == pytest.approx(1.0) and rec["in_band"] is True
    assert rec["pressure_step"] == 0.02 and rec["report_only"] is True
    assert rec["total_nominal"] == 2e5 and rec["total_installed"] == 199_219.0
    assert rec["excluded_coils"] == ["F9A", "F9B"]
    assert ("pressure", pytest.approx(1.02e4)) in calls
    # restored: the delivered equilibrium, then the reconstruction's reg
    assert calls[-2:] == [("replace_eq", "DELIVERED"), ("set_coil_reg", 1)]


def test_a_partition_outside_the_band_is_warned_not_acted_on(monkeypatch):
    bq, Backend, calls = _fake_case(monkeypatch, coil_shift=0.002, rms=5.0)
    with pytest.warns(UserWarning, match="outside"):
        rec = BB.measure_partition(
            bq, np.ones(3), np.full(3, 1e4), lambda maxits: Backend(),
            psi_pad=1e-3, maxits=100, coil_reg_terms=["RECON"],
            geom_record=dict(isoflux_weight_total=2e5))
    assert rec["partition"] == pytest.approx(10.0) and rec["in_band"] is False


def test_a_failed_probe_solve_is_recorded_and_restored(monkeypatch):
    bq, Backend, calls = _fake_case(monkeypatch, solve_raises=True)
    with pytest.warns(UserWarning, match="not measured"):
        rec = BB.measure_partition(
            bq, np.ones(3), np.full(3, 1e4), lambda maxits: Backend(),
            psi_pad=1e-3, maxits=100, coil_reg_terms=["RECON"],
            geom_record=dict(isoflux_weight_total=2e5))
    assert rec["status"] == "probe_failed" and "maxits" in rec["error"]
    assert calls[-2:] == [("replace_eq", "DELIVERED"), ("set_coil_reg", 1)]


def test_the_probe_is_skipped_when_it_could_not_be_undone(monkeypatch):
    bq, Backend, calls = _fake_case(monkeypatch)
    with pytest.warns(UserWarning, match="not on record"):
        rec = BB.measure_partition(
            bq, np.ones(3), np.full(3, 1e4), lambda maxits: Backend(),
            psi_pad=1e-3, maxits=100, coil_reg_terms=None,
            geom_record=dict(isoflux_weight_total=2e5))
    assert rec["n_solves"] == 0 and not calls
    bq.mygs = object()                          # a stand-in: not probed
    with pytest.warns(UserWarning, match="cannot be probed"):
        rec = BB.measure_partition(
            bq, np.ones(3), np.full(3, 1e4), lambda maxits: Backend(),
            psi_pad=1e-3, maxits=100, coil_reg_terms=["RECON"],
            geom_record={})
    assert rec["n_solves"] == 0


def test_the_probe_switch():
    from bouquet.config import SolverConfig
    from bouquet.engine import partition_probe_on
    assert SolverConfig(mesh_path="m").boundary_partition_probe is True
    geom = types.SimpleNamespace(iso_pts=np.zeros((4, 2)))
    bq = types.SimpleNamespace(_geom=geom, config=types.SimpleNamespace(
        solver=SolverConfig(mesh_path="m")))
    assert partition_probe_on(bq)
    bq.config.solver.boundary_partition_probe = False
    assert not partition_probe_on(bq)
    assert BB.partition_warning(dict(status="off")) is None


# ---------------------------------------------------------------------------
#  the geometry: constant, override -- and the stamps
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


def test_the_default_geometry_installs_the_constant(monkeypatch):
    g = _geom(monkeypatch)
    rec = g.record()
    assert rec["isoflux_weight_mode"] == "constant"
    assert rec["isoflux_weight_total"] == pytest.approx(SG.ISOFLUX_WEIGHT_TOTAL)
    assert rec["isoflux_balance"]["source"] == \
        "solver_geometry.ISOFLUX_WEIGHT_TOTAL"


def test_an_override_is_installed_as_given(monkeypatch):
    g = _geom(monkeypatch, override=5e4)
    rec = g.record()
    assert rec["isoflux_weight_mode"] == "override"
    assert rec["isoflux_weight_total"] == pytest.approx(5e4)
    assert rec["isoflux_balance"]["source"] == "SolverConfig.isoflux_weight_total"


def test_the_partition_stamp_keeps_the_set(monkeypatch):
    g = _geom(monkeypatch)
    part = dict(partition=0.97, in_band=True, status="measured",
                n_solves=1, wall_s=12.0)
    c = SG.with_partition(g, part)
    assert np.array_equal(c.iso_pts, g.iso_pts)
    assert np.array_equal(c.iso_w, g.iso_w)
    rec = c.record()
    assert rec["isoflux_sha256"] == g.record()["isoflux_sha256"]
    assert rec["isoflux_weight_mode"] == "constant"
    assert rec["isoflux_balance"]["partition"]["partition"] == 0.97
    assert rec["isoflux_weight_installed_total"] == pytest.approx(
        SG.ISOFLUX_WEIGHT_TOTAL - g.iso_w[0])


def test_the_override_field_exists_and_defaults_to_the_constant():
    from bouquet.config import SolverConfig
    assert SolverConfig(mesh_path="m").isoflux_weight_total is None
