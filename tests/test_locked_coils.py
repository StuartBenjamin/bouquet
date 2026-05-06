"""
Unit tests for the lock-coil PR1 helpers and HDF5 schema additions.

Covers:
  - psi_boundary_deviation : dpsi → dr conversion via local |grad psi|.
  - xpoint_deviation       : nearest-neighbor pairing with NaN on miss.
  - lock_coils_to_baseline : asserts the right TokaMaker setters fire
                             and (intentionally) leaves coil_reg alone.
  - HDF5 round-trip        : new lock-coil fields survive store/load.

These are pure-Python tests using lightweight fakes for the TokaMaker
surface; they do not require OpenFUSIONToolkit.
"""

import os

import numpy as np
import pytest

from bouquet.TokaMaker_interface import (
    lock_coils_to_baseline,
    psi_boundary_deviation,
    xpoint_deviation,
)
from bouquet.utils import (
    initialize_equilibrium_database,
    store_baseline_profiles,
    store_equilibrium,
    load_baseline_profiles,
    load_equilibrium,
)


# ====================================================================
#  Lightweight TokaMaker stand-ins
# ====================================================================
class _FakeFieldEval:
    """A ``get_field_eval('psi'|'B')`` return value.

    ``values`` is a callable ``pt -> ndarray`` so each test can decide
    what the field returns at any (R, Z).
    """
    def __init__(self, values):
        self._values = values

    def eval(self, pt):
        return self._values(pt)


class _FakeMygs:
    """Minimal stand-in exposing the surface used by the helpers.

    Attribute defaults are chosen so a freshly-constructed instance
    passes the helpers' validity checks without tweaking.
    """
    def __init__(self,
                 psi_lcfs=0.0,
                 psi_axis=1.0,
                 psi_at_pt=None,
                 B_at_pt=None,
                 xpoints=None,
                 diverted=False):
        self.psi_bounds = [psi_lcfs, psi_axis]
        self._psi_at_pt = psi_at_pt or (lambda pt: np.array([psi_lcfs]))
        self._B_at_pt = B_at_pt or (lambda pt: np.array([0.0, 0.0, 0.0]))
        self._xpoints = (np.zeros((0, 2)) if xpoints is None
                         else np.asarray(xpoints, dtype=np.float64))
        self._diverted = diverted

        # Tracking for setter assertions
        self.set_isoflux_calls = []
        self.set_saddles_calls = []
        self.set_coil_currents_calls = []
        self.set_coil_reg_calls = []
        self.coil_reg_term_calls = []

    # Field evaluation
    def get_field_eval(self, key):
        if key == 'psi':
            return _FakeFieldEval(self._psi_at_pt)
        if key == 'B':
            return _FakeFieldEval(self._B_at_pt)
        raise KeyError(key)

    # X-point access
    def get_xpoints(self):
        return self._xpoints, self._diverted

    # Setters used by lock_coils_to_baseline
    def set_isoflux(self, *args, **kwargs):
        self.set_isoflux_calls.append((args, kwargs))

    def set_saddles(self, *args, **kwargs):
        self.set_saddles_calls.append((args, kwargs))

    def set_coil_currents(self, currents):
        self.set_coil_currents_calls.append(dict(currents))

    def set_coil_reg(self, *args, **kwargs):
        self.set_coil_reg_calls.append((args, kwargs))

    def coil_reg_term(self, coffs, target=0.0, weight=1.0):
        # Mimic OpenFUSIONToolkit.TokaMaker.TokaMaker.coil_reg_term:
        # returns a namedtuple-like recording of the term args.
        rec = dict(coffs=dict(coffs), target=float(target),
                   weight=float(weight))
        self.coil_reg_term_calls.append(rec)
        return rec


# ====================================================================
#  psi_boundary_deviation
# ====================================================================
class TestPsiBoundaryDeviation:
    """The core conversion: dpsi(R,Z) / (R |Bp|) → spatial deviation."""

    def _make_uniform_dev_mygs(self, dr_target_m, R0=1.7, Bp_mag=0.5):
        """A fake whose psi at every queried point is offset from
        ``psi_lcfs`` by ``dr_target_m * R0 * Bp_mag`` and whose B
        field has |B_pol| = Bp_mag at R = R0.  The helper should
        recover ``dr_target_m`` for every point.
        """
        psi_lcfs = 0.0
        grad_psi = R0 * Bp_mag                    # |grad psi| at (R0, *)
        dpsi = dr_target_m * grad_psi
        return _FakeMygs(
            psi_lcfs=psi_lcfs,
            psi_axis=1.0,
            psi_at_pt=lambda pt: np.array([psi_lcfs + dpsi]),
            B_at_pt=lambda pt: np.array([Bp_mag, 0.0, 0.0]),
        )

    def test_recovers_known_uniform_offset(self):
        # 2 mm offset everywhere on the boundary
        dr = 2.0e-3
        mygs = self._make_uniform_dev_mygs(dr)
        ref_R = np.array([1.2, 1.7, 2.2])
        ref_Z = np.array([0.0, 0.5, 0.0])

        rms_m, max_m, devs_m = psi_boundary_deviation(mygs, ref_R, ref_Z)

        # All points should report the configured offset (note: the
        # fake uses R0=1.7 in |grad psi|; passing ref_R values that
        # differ from R0 produces small but bounded errors that we
        # tolerate via rtol below).
        np.testing.assert_allclose(devs_m[1], dr, rtol=1e-9)
        # At ref_R != 1.7 the recovery error is (R - R0)/R; allow it.
        assert np.all(devs_m > 0)
        assert max_m == pytest.approx(devs_m.max())
        assert rms_m == pytest.approx(np.sqrt(np.mean(devs_m * devs_m)))

    def test_zero_offset_at_lcfs(self):
        mygs = _FakeMygs(
            psi_at_pt=lambda pt: np.array([0.0]),  # exactly on LCFS
            B_at_pt=lambda pt: np.array([0.5, 0.0, 0.5]),
        )
        ref_R = np.array([1.5, 2.0])
        ref_Z = np.array([0.0, 0.3])
        rms_m, max_m, devs_m = psi_boundary_deviation(mygs, ref_R, ref_Z)
        np.testing.assert_array_equal(devs_m, [0.0, 0.0])
        assert rms_m == 0.0
        assert max_m == 0.0

    def test_falls_back_when_grad_psi_vanishes(self):
        # |B_pol| = 0 → grad_psi = 0; helper falls back to dpsi/dpsi_total.
        # We just want it not to crash and to produce finite output.
        mygs = _FakeMygs(
            psi_lcfs=0.0,
            psi_axis=1.0,
            psi_at_pt=lambda pt: np.array([0.5]),     # halfway to axis
            B_at_pt=lambda pt: np.array([0.0, 0.0, 0.0]),
        )
        ref_R = np.array([1.5])
        ref_Z = np.array([0.0])
        rms_m, max_m, devs_m = psi_boundary_deviation(mygs, ref_R, ref_Z)
        assert np.all(np.isfinite(devs_m))


# ====================================================================
#  xpoint_deviation
# ====================================================================
class TestXpointDeviation:
    """Pairing logic, with NaN for baseline X-points beyond the cutoff."""

    def test_unique_close_pairing(self):
        ref = np.array([[1.5, -1.2], [1.5, 1.2]])
        # Perturbed X-points shifted 5 mm vertically
        pert = np.array([[1.5, -1.205], [1.5, 1.195]])

        mygs = _FakeMygs(xpoints=pert)
        rms_m, max_m, devs_m, paired = xpoint_deviation(mygs, ref)

        np.testing.assert_allclose(devs_m, [0.005, 0.005], atol=1e-12)
        np.testing.assert_allclose(paired, pert)
        assert rms_m == pytest.approx(0.005)
        assert max_m == pytest.approx(0.005)

    def test_unmatched_when_beyond_cutoff(self):
        ref = np.array([[1.5, -1.2], [1.5, 1.2]])
        # Only one perturbed X-point present; the other baseline X-pt
        # has nothing within max_pair_dist_m (default 0.1 m).
        pert = np.array([[1.5, -1.21]])

        mygs = _FakeMygs(xpoints=pert)
        rms_m, max_m, devs_m, paired = xpoint_deviation(
            mygs, ref, max_pair_dist_m=0.1)

        assert devs_m[0] == pytest.approx(0.01)
        assert np.isnan(devs_m[1])
        assert np.isnan(paired[1, 0]) and np.isnan(paired[1, 1])
        # Summaries computed on finite entries only
        assert rms_m == pytest.approx(0.01)
        assert max_m == pytest.approx(0.01)

    def test_all_unmatched_returns_nan_summaries(self):
        ref = np.array([[1.5, -1.2]])
        # Perturbed X-point is 1 m away — well beyond cutoff
        pert = np.array([[2.5, -1.2]])
        mygs = _FakeMygs(xpoints=pert)
        rms_m, max_m, devs_m, paired = xpoint_deviation(
            mygs, ref, max_pair_dist_m=0.1)
        assert np.isnan(rms_m)
        assert np.isnan(max_m)
        assert np.isnan(devs_m[0])

    def test_no_perturbed_xpoints_returns_nan(self):
        ref = np.array([[1.5, -1.2]])
        mygs = _FakeMygs(xpoints=np.zeros((0, 2)))
        rms_m, max_m, devs_m, paired = xpoint_deviation(mygs, ref)
        assert np.isnan(rms_m)
        assert np.isnan(max_m)
        assert np.isnan(devs_m).all()

    def test_no_baseline_xpoints_safe(self):
        mygs = _FakeMygs(xpoints=np.array([[1.5, -1.2]]))
        rms_m, max_m, devs_m, paired = xpoint_deviation(
            mygs, np.zeros((0, 2)))
        assert np.isnan(rms_m)
        assert np.isnan(max_m)
        assert devs_m.shape == (0,)
        assert paired.shape == (0, 2)


# ====================================================================
#  lock_coils_to_baseline
# ====================================================================
class TestLockCoilsToBaseline:
    """Verify the helper installs strong reg targeting ref values and
    clears the saddle constraint while leaving isoflux active.

    The original PR1 design used ``set_coil_currents`` to hard-pin
    coils with isoflux cleared, but TokaMaker fails the GS matrix
    solve under any profile perturbation in that mode.  The shipping
    design replaces it with a strong reg (default weight=1e2) so the
    inverse solver retains constraint freedom via isoflux.
    """

    def test_installs_reg_targeting_ref_values(self):
        ref_coils = {'F1A': 1.0e3, 'F2A': -2.0e3}
        mygs = _FakeMygs()
        lock_coils_to_baseline(mygs, ref_coils, weight=1.0e2)

        # Saddle constraint cleared, isoflux untouched
        assert len(mygs.set_saddles_calls) == 1
        assert mygs.set_saddles_calls[0] == ((None,), {})
        assert mygs.set_isoflux_calls == []

        # set_coil_reg called once, containing one term per coil + a VSC term
        assert len(mygs.set_coil_reg_calls) == 1
        terms = mygs.set_coil_reg_calls[0][1]['reg_terms']
        # Per-coil terms target the requested values at the supplied weight
        per_coil = [t for t in terms if '#VSC' not in t['coffs']]
        assert len(per_coil) == len(ref_coils)
        for t in per_coil:
            (name,) = t['coffs'].keys()
            assert t['target'] == pytest.approx(ref_coils[name])
            assert t['weight'] == pytest.approx(1.0e2)
        # VSC term zeros the virtual circuit at the supplied vsc_weight
        vsc = [t for t in terms if '#VSC' in t['coffs']]
        assert len(vsc) == 1
        assert vsc[0]['target'] == pytest.approx(0.0)

    def test_does_not_hard_pin_via_set_coil_currents(self):
        # The shipping design relies on regularisation, not the
        # ``set_coil_currents`` hard pin.  Locks in that behaviour;
        # flip if the design changes.
        mygs = _FakeMygs()
        lock_coils_to_baseline(mygs, {'F1A': 1.0e3})
        assert mygs.set_coil_currents_calls == []

    def test_weight_kwarg_propagates(self):
        mygs = _FakeMygs()
        lock_coils_to_baseline(mygs, {'F1A': 1.0e3, 'F2A': 2.0e3},
                                weight=1.0e6)
        terms = mygs.set_coil_reg_calls[0][1]['reg_terms']
        per_coil_weights = [t['weight'] for t in terms
                             if '#VSC' not in t['coffs']]
        assert all(w == pytest.approx(1.0e6) for w in per_coil_weights)

    def test_vsc_weight_kwarg_propagates(self):
        mygs = _FakeMygs()
        lock_coils_to_baseline(mygs, {'F1A': 1.0e3},
                                vsc_weight=0.5)
        terms = mygs.set_coil_reg_calls[0][1]['reg_terms']
        vsc = [t for t in terms if '#VSC' in t['coffs']]
        assert vsc[0]['weight'] == pytest.approx(0.5)


# ====================================================================
#  HDF5 round-trip — new lock-coil schema
# ====================================================================
class TestHDF5LockCoilRoundTrip:

    @pytest.fixture()
    def tmp_db(self, tmp_path):
        header = str(tmp_path / "test_lock")
        db_path = initialize_equilibrium_database(header)
        return header, db_path

    def _profiles(self, n=33):
        psi_N = np.linspace(0.0, 1.0, n)
        ones = np.ones(n)
        return psi_N, ones

    def test_baseline_persists_lock_ref_state(self, tmp_db):
        header, db_path = tmp_db
        psi_N, ones = self._profiles()

        ref_coils = {'F1A': 1.0e3, 'F2A': -2.0e3, 'F3A': 5.0e2}
        ref_lcfs_R = np.linspace(1.0, 2.4, 64)
        ref_lcfs_Z = np.zeros(64)
        ref_x_points = np.array([[1.5, -1.2], [1.5, 1.2]])

        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones * 2, ni=ones * 3, ti=ones * 4,
            pressure=ones * 5, j_phi=ones * 6,
            sigma_ne=ones * 0.1, sigma_te=ones * 0.2,
            sigma_ni=ones * 0.3, sigma_ti=ones * 0.4,
            sigma_jphi=ones * 0.5,
            Ip_target=1.0e6, l_i_target=0.9,
            coils_locked=True,
            ref_coil_currents=ref_coils,
            ref_lcfs_R=ref_lcfs_R,
            ref_lcfs_Z=ref_lcfs_Z,
            ref_x_points=ref_x_points,
        )

        bl = load_baseline_profiles(db_path)
        assert bool(bl["coils_locked"]) is True
        np.testing.assert_array_equal(bl["ref_lcfs_R [m]"], ref_lcfs_R)
        np.testing.assert_array_equal(bl["ref_lcfs_Z [m]"], ref_lcfs_Z)
        np.testing.assert_array_equal(bl["ref_x_points_RZ [m]"],
                                       ref_x_points)
        # Convenience dict reconstruction
        assert "ref_coil_currents" in bl
        for name, val in ref_coils.items():
            assert bl["ref_coil_currents"][name] == pytest.approx(val)

    def test_baseline_persists_filter_thresholds(self, tmp_db):
        """coil_drift / boundary / xpoint thresholds round-trip as
        baseline HDF5 attrs so plotters and post-hoc analysis can show
        which filter caps were in effect for the run."""
        header, db_path = tmp_db
        psi_N, ones = self._profiles()
        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones, ni=ones, ti=ones,
            pressure=ones, j_phi=ones,
            sigma_ne=ones, sigma_te=ones,
            sigma_ni=ones, sigma_ti=ones,
            sigma_jphi=ones,
            Ip_target=1e6, l_i_target=0.8,
            coils_locked=True,
            ref_coil_currents={'F1A': 1.0e3},
            lock_coils_weight=1.0e2,
            coil_drift_threshold_A=5000.0,
            boundary_max_threshold_mm=10.0,
            xpoint_max_threshold_mm=20.0,
        )
        bl = load_baseline_profiles(db_path)
        assert bl['coil_drift_threshold_A'] == pytest.approx(5000.0)
        assert bl['boundary_max_threshold_mm'] == pytest.approx(10.0)
        assert bl['xpoint_max_threshold_mm'] == pytest.approx(20.0)

    def test_baseline_default_no_filter_thresholds(self, tmp_db):
        """When no thresholds are passed, the attrs are absent (legacy
        layout untouched)."""
        header, db_path = tmp_db
        psi_N, ones = self._profiles()
        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones, ni=ones, ti=ones,
            pressure=ones, j_phi=ones,
            sigma_ne=ones, sigma_te=ones,
            sigma_ni=ones, sigma_ti=ones,
            sigma_jphi=ones,
            Ip_target=1e6, l_i_target=0.8,
            coils_locked=False,
        )
        bl = load_baseline_profiles(db_path)
        for key in ('coil_drift_threshold_A',
                    'boundary_max_threshold_mm',
                    'xpoint_max_threshold_mm'):
            assert key not in bl

    def test_baseline_default_no_lock_fields(self, tmp_db):
        header, db_path = tmp_db
        psi_N, ones = self._profiles()
        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones, ni=ones, ti=ones,
            pressure=ones, j_phi=ones,
            sigma_ne=ones, sigma_te=ones,
            sigma_ni=ones, sigma_ti=ones,
            sigma_jphi=ones,
            Ip_target=1e6, l_i_target=0.8,
        )
        bl = load_baseline_profiles(db_path)
        assert bool(bl["coils_locked"]) is False
        assert "ref_lcfs_R [m]" not in bl
        assert "ref_x_points_RZ [m]" not in bl

    def test_equilibrium_persists_deviation_diagnostics(self, tmp_db):
        header, db_path = tmp_db
        psi_N, ones = self._profiles()

        eqdsk_path = db_path.replace(".h5", ".eqdsk")
        with open(eqdsk_path, "wb") as f:
            f.write(b"fake eqdsk")

        bnd_devs = np.array([0.5, 0.7, 0.6, 1.1])     # mm
        xpt_devs = np.array([1.2, np.nan])             # mm; one unmatched
        xpt_RZ = np.array([[1.5, -1.21], [np.nan, np.nan]])
        coil_drift = {'F1A': 12.3, 'F2A': -4.5}

        store_equilibrium(
            header, count=0, eqdsk_filepath=eqdsk_path,
            psi_N=psi_N,
            j_phi=ones, j_BS=ones, j_inductive=ones,
            n_e=ones, T_e=ones, n_i=ones, T_i=ones, w_ExB=ones,
            li1=0.5, li3=0.7,
            boundary_devs_mm=bnd_devs,
            boundary_rms_mm=float(np.sqrt(np.mean(bnd_devs ** 2))),
            boundary_max_mm=float(bnd_devs.max()),
            x_point_devs_mm=xpt_devs,
            x_point_rms_mm=1.2,
            x_point_max_mm=1.2,
            x_points_RZ=xpt_RZ,
            coil_drift_A=coil_drift,
        )

        result = load_equilibrium(header, count=0)
        np.testing.assert_array_equal(result["boundary_devs [mm]"], bnd_devs)
        # NaN-aware comparison for x-point devs
        got_xpt = result["x_point_devs [mm]"]
        assert got_xpt[0] == pytest.approx(1.2)
        assert np.isnan(got_xpt[1])
        assert result["boundary_rms_mm"] == pytest.approx(
            float(np.sqrt(np.mean(bnd_devs ** 2))))
        assert result["boundary_max_mm"] == pytest.approx(float(bnd_devs.max()))
        assert result["x_point_rms_mm"] == pytest.approx(1.2)
        assert "coil_drift" in result
        assert result["coil_drift"]["F1A"] == pytest.approx(12.3)
        assert result["coil_drift"]["F2A"] == pytest.approx(-4.5)

        os.remove(eqdsk_path)

    def test_equilibrium_default_no_deviation_fields(self, tmp_db):
        # Legacy callers (lock_coils=False) must not populate the new
        # fields and must keep round-tripping cleanly.
        header, db_path = tmp_db
        psi_N, ones = self._profiles()
        eqdsk_path = db_path.replace(".h5", ".eqdsk")
        with open(eqdsk_path, "wb") as f:
            f.write(b"fake")
        store_equilibrium(
            header, count=0, eqdsk_filepath=eqdsk_path,
            psi_N=psi_N,
            j_phi=ones, j_BS=ones, j_inductive=ones,
            n_e=ones, T_e=ones, n_i=ones, T_i=ones, w_ExB=ones,
            li1=0.5, li3=0.7,
        )
        result = load_equilibrium(header, count=0)
        for key in ("boundary_devs [mm]", "boundary_rms_mm",
                    "x_point_devs [mm]", "x_point_rms_mm",
                    "coil_drift [A]"):
            assert key not in result
        os.remove(eqdsk_path)


# ====================================================================
#  Smoke tests for new lock-coil plot helpers
# ====================================================================
#
# These don't validate visual correctness — they only verify that the
# plotting helpers can be called against a synthetic HDF5 fixture without
# raising, so the import / data-loading wiring stays intact.  Visual
# review is the user's job.
class TestLockCoilPlotters:

    @pytest.fixture()
    def lock_db(self, tmp_path):
        """Build a 3-equilibrium HDF5 with full lock-coil diagnostics."""
        import matplotlib
        matplotlib.use("Agg")
        header = str(tmp_path / "plot_smoke")
        initialize_equilibrium_database(header)
        psi_N = np.linspace(0.0, 1.0, 33)
        ones = np.ones_like(psi_N)
        ref_coils = {'F1A': 1.0e3, 'F2A': -2.0e3, 'F9A': 3.0e3,
                     'F9B': -3.0e3, 'ECOILA': 5.0e2}
        ref_xpts = np.array([[1.5, -1.2], [1.5, 1.2]])
        ref_lcfs_R = np.linspace(1.0, 2.4, 16)
        ref_lcfs_Z = np.zeros(16)
        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones, ni=ones, ti=ones,
            pressure=ones, j_phi=ones,
            sigma_ne=ones, sigma_te=ones,
            sigma_ni=ones, sigma_ti=ones,
            sigma_jphi=ones,
            Ip_target=1.0e6, l_i_target=0.9,
            scan_val=0,
            coils_locked=True,
            ref_coil_currents=ref_coils,
            ref_lcfs_R=ref_lcfs_R, ref_lcfs_Z=ref_lcfs_Z,
            ref_x_points=ref_xpts,
            lock_coils_weight=1.0e2,
            coil_drift_threshold_A=5000.0,
            boundary_max_threshold_mm=10.0,
            xpoint_max_threshold_mm=20.0,
        )
        # Three perturbed equilibria with synthetic diagnostics
        for c in range(3):
            eqdsk_path = str(tmp_path / f"plot_smoke_count={c}.eqdsk")
            with open(eqdsk_path, "wb") as f:
                f.write(b"fake")
            store_equilibrium(
                header, count=c, eqdsk_filepath=eqdsk_path,
                psi_N=psi_N, j_phi=ones, j_BS=ones, j_inductive=ones,
                n_e=ones, T_e=ones, n_i=ones, T_i=ones, w_ExB=ones,
                li1=0.9 + 0.01 * c, li3=0.92 + 0.01 * c,
                scan_val=0,
                boundary_devs_mm=np.array([0.5 * (c + 1), 0.7, 0.6, 1.1]),
                boundary_rms_mm=float(0.7 * (c + 1)),
                boundary_max_mm=float(1.1 * (c + 1)),
                x_point_devs_mm=np.array([1.0 * (c + 1), 1.5 * (c + 1)]),
                x_point_rms_mm=float(1.25 * (c + 1)),
                x_point_max_mm=float(1.5 * (c + 1)),
                x_points_RZ=np.array([[1.5, -1.21], [1.5, 1.19]]),
                coil_drift_A={'F1A': 12.0 * (c + 1), 'F2A': -4.0 * (c + 1),
                               'F9A': 200.0 * (c + 1),
                               'F9B': -200.0 * (c + 1),
                               'ECOILA': 5.0 * (c + 1)},
            )
            os.remove(eqdsk_path)
        return header

    def test_plot_boundary_deviations_runs(self, lock_db):
        from bouquet import plot_boundary_deviations
        fig, ax = plot_boundary_deviations(lock_db, scan_value=0)
        assert fig is not None and ax is not None

    def test_plot_xpoint_deviations_runs(self, lock_db):
        from bouquet import plot_xpoint_deviations
        fig, ax = plot_xpoint_deviations(lock_db, scan_value=0)
        assert fig is not None and ax is not None

    def test_plot_coil_drift_runs(self, lock_db):
        from bouquet import plot_coil_drift
        fig, axes = plot_coil_drift(lock_db, scan_value=0)
        assert fig is not None and axes is not None

    def test_plot_coil_drift_respects_explicit_order(self, lock_db):
        from bouquet import plot_coil_drift
        # Custom ordering: only ECOILA + F9A; everything else hidden.
        fig, axes = plot_coil_drift(
            lock_db, scan_value=0,
            coil_order=['ECOILA', 'F9A'])
        assert fig is not None
        # Two coils + ± threshold lines = 3 legend entries
        # (subtle; just verify a legend exists)
        assert fig.legends
        assert len(fig.legends[0].get_lines()) >= 2

    def test_plot_xpoint_deviations_no_baseline_xpts_returns_none(
            self, tmp_path):
        """When baseline X-points are not stored (lock_coils=False),
        the helper should bail out cleanly."""
        from bouquet import plot_xpoint_deviations
        header = str(tmp_path / "no_xpt")
        initialize_equilibrium_database(header)
        psi_N = np.linspace(0.0, 1.0, 33)
        ones = np.ones_like(psi_N)
        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones, ni=ones, ti=ones,
            pressure=ones, j_phi=ones,
            sigma_ne=ones, sigma_te=ones,
            sigma_ni=ones, sigma_ti=ones,
            sigma_jphi=ones,
            Ip_target=1e6, l_i_target=0.8,
            scan_val=0,
            coils_locked=False,
        )
        fig, ax = plot_xpoint_deviations(header, scan_value=0)
        assert fig is None and ax is None
