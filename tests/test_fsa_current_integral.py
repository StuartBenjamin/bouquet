"""The flux-surface-averaged plasma-current measure (``utils.Ip_fsa_integral``).

Two halves.

**Fast** (pure NumPy): the algebra of :func:`bouquet.utils.Ip_fsa_weights` --
that the measure is affine with the ``P'`` term in the constant, that the
``fsa`` reading has no constant, that a circular large-aspect-ratio geometry
integrates to the analytic answer, and that every unsupported combination
raises instead of returning a plausible number.

**Solver** (``pytest -m solver``, in a subprocess): the validation the measure
was accepted on -- integrate the SOLVED equilibrium's OWN current profile and
compare with ``compute_area_integral(calc_jtor_plasma)``.  Required: 0.1 %.
Measured on the synthetic D3D-like baseline: **+0.0071 %** in both conventions.
The same run pins the three properties the implementation depends on:

  * ``get_q(psi=...)`` collapses SILENTLY onto the magnetic axis if the sample
    grid includes ``psi_N = 0`` -- ``<R>`` constant to 2e-15 across all 257
    surfaces, ``dV/dPsi`` constant, no exception.  ``fsa_current_geometry``
    clips the grid and asserts against the collapse;
  * ``dV/dPsi`` is per DIMENSIONAL psi (``int dV/dPsi dpsi`` recovers the
    volume to -0.25 %; the ``dpsi_N`` reading is out by +291 %);
  * ``compute_flux_integral`` is NOT ``int_plasma f dA``.  It covers the whole
    limiter region with the profile pinned at its LCFS value outside the
    plasma, so ``FI(1) = 2.8385 m^2`` against a true plasma cross-section of
    ``1.7901 m^2``.  That is where 7dc254b's "+12.9 % convention bias" came
    from, and it is why the fix is a measure rather than a calibration.

Runs on the synthetic D3D-like example (no proprietary data).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from bouquet.utils import Ip_fsa_integral, Ip_fsa_weights

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))

#: The step-1 acceptance the measure had to clear before being wired in.
#: Measured +0.0071 % -- 14x margin.  Not to be widened: a miss means the
#: V'/<1/R> plumbing or the psi_N -> psi Jacobian has moved.
_SELF_CONSISTENCY = 1.0e-3


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  fast: the algebra, on a synthetic geometry
# ---------------------------------------------------------------------------
def _circular_geom(npsi=401, R0=3.0, a=0.3):
    """Large-aspect-ratio circular flux surfaces with a known area element.

    ``r = a*sqrt(psi_N)`` (so ``dA/dpsi_N = pi a^2`` is constant) and the FSA
    of any function of R is taken at R0 -- exact in the ``a/R0 -> 0`` limit,
    which is enough to give the current integral a closed form.
    """
    psi_N = np.linspace(0.0, 1.0, npsi)
    ones = np.ones_like(psi_N)
    dpsi_dpsiN = 2.0
    inv_R = ones / R0
    # dA/dpsi_N = (V'/2pi) <1/R> |dpsi/dpsi_N| = pi a^2  =>  V' = 2 pi^2 a^2 R0
    dV_dpsi = (np.pi * a ** 2) * (2.0 * np.pi) * R0 / dpsi_dpsiN * ones
    return {
        "psi_N": psi_N, "psi_q": psi_N,
        "R_avg": R0 * ones, "inv_R": inv_R, "inv_R2": ones / R0 ** 2,
        "dV_dpsi": dV_dpsi, "dpsi_dpsiN": dpsi_dpsiN,
        "pprime": np.zeros_like(psi_N),
        "dA_dpsiN": dV_dpsi / (2.0 * np.pi) * inv_R * dpsi_dpsiN,
    }


def test_uniform_current_integrates_to_j_times_area():
    """A flat profile over a known cross-section: I_p = J * A, both readings."""
    geom = _circular_geom()
    area = float(np.pi * 0.3 ** 2)
    J = np.full_like(geom["psi_N"], 1.5e6)
    for convention in ("fsa", "jphi-linterp"):
        got = Ip_fsa_integral(None, geom["psi_N"], J, convention=convention,
                              geom=geom)
        assert got == pytest.approx(1.5e6 * area, rel=1e-12), convention


def test_fsa_reading_has_no_constant_term_and_jphi_reading_does():
    """The P' term is independent of J, so it must live in ``c``, not ``w``."""
    geom = _circular_geom()
    _w, c = Ip_fsa_weights(geom, convention="fsa")
    assert c == 0.0
    geom_p = dict(geom, pprime=np.full_like(geom["psi_N"], 2.0e5))
    # <R><1/R^2>/<1/R> == 1 exactly in this geometry, so even a finite P'
    # contributes nothing -- the correction is a shaping effect.
    _w, c = Ip_fsa_weights(geom_p, convention="jphi-linterp")
    assert c == pytest.approx(0.0, abs=1e-6)
    # ... but bend <1/R^2> away from <1/R>^2 and it must show up.
    geom_s = dict(geom_p, inv_R2=geom["inv_R2"] * 1.1)
    _w, c = Ip_fsa_weights(geom_s, convention="jphi-linterp")
    assert abs(c) > 1.0e3


def test_the_measure_is_affine_in_the_profile():
    """What ``_AnchorIpRenorm.solve_scale`` roots analytically instead of by
    bisection.  If this ever stops holding the analytic root is wrong."""
    geom = dict(_circular_geom(), pprime=np.full_like(_circular_geom()["psi_N"],
                                                      2.0e5))
    geom["inv_R2"] = geom["inv_R2"] * 1.07
    rng = np.random.default_rng(7)
    j1 = 1e6 * (1.0 - geom["psi_N"] ** 2) + 1e4 * rng.standard_normal(geom["psi_N"].size)
    j2 = 3e5 * np.exp(-((geom["psi_N"] - 0.95) / 0.03) ** 2)
    _w, c = Ip_fsa_weights(geom, convention="jphi-linterp")
    ip = lambda p: Ip_fsa_integral(None, geom["psi_N"], p,  # noqa: E731
                                   convention="jphi-linterp", geom=geom)
    for a in (0.5, 1.0, 2.75):
        assert ip(a * j1 + j2) == pytest.approx(a * (ip(j1) - c) + ip(j2),
                                                rel=1e-12)


def test_unsupported_combinations_raise_rather_than_guess():
    geom = _circular_geom()
    with pytest.raises(ValueError, match="unknown convention"):
        Ip_fsa_weights(geom, convention="area")
    with pytest.raises(ValueError, match="<1/R\\^2>"):
        Ip_fsa_weights(dict(geom, inv_R2=None), convention="jphi-linterp")
    with pytest.raises(ValueError, match="P'"):
        Ip_fsa_weights(dict(geom, pprime=None), convention="jphi-linterp")


def test_r2_mode_resolution():
    """The A/B switch, including 7dc254b's spelling of the ratio mode."""
    from bouquet.TokaMaker_interface import _r2_ip_mode, _R2_IP_MODE_DEFAULT

    saved = os.environ.pop("BOUQUET_R2_IP_MODE", None)
    try:
        assert _r2_ip_mode() == _R2_IP_MODE_DEFAULT
        for given, want in (("exact", "exact"), ("fsa", "fsa"),
                            ("ratio", "ratio"), ("anchor", "ratio"),
                            ("ANCHOR", "ratio"), (" legacy ", "legacy")):
            os.environ["BOUQUET_R2_IP_MODE"] = given
            assert _r2_ip_mode() == want, given
        os.environ["BOUQUET_R2_IP_MODE"] = "exakt"
        with pytest.raises(ValueError, match="BOUQUET_R2_IP_MODE"):
            _r2_ip_mode()
    finally:
        os.environ.pop("BOUQUET_R2_IP_MODE", None)
        if saved is not None:
            os.environ["BOUQUET_R2_IP_MODE"] = saved


# ---------------------------------------------------------------------------
#  solver: the validation the measure was accepted on
# ---------------------------------------------------------------------------
def _probe(outdir):
    """Everything the measure is validated by, on the solved D3D-like anchor.

    Subprocess entry point (``OFT_env`` is a per-process singleton, so this
    module must never build a solver in the pytest process).  Results land in
    ``<outdir>/fsa.json``.
    """
    import numpy as np
    import bouquet as bq
    from bouquet.utils import (fsa_current_geometry, Ip_fsa_integral,
                               eq_jphi_profile)

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "fsa"), n_draws=1)
    b.setup_solver()
    bl = b.prepare_baseline()
    mygs = b.mygs
    psi_N = np.asarray(bl.psi_N, dtype=float)
    J = np.asarray(bl.j_phi, dtype=float)
    Ip_true = float(mygs.compute_area_integral(mygs.calc_jtor_plasma()))
    vol_true = float(mygs.get_stats(
        lcfs_pad=float(getattr(b.config.source, "psi_pad", 1e-3)))["vol"])

    out = {"Ip_true": Ip_true, "vol_true": vol_true,
           "flux_integral_of_one": float(
               mygs.compute_flux_integral(psi_N, np.ones_like(psi_N)))}

    snap = mygs.copy_eq()
    for tag, obj in (("live", mygs), ("snapshot", snap)):
        geom = fsa_current_geometry(obj, psi_N)
        sgn = 1.0 if float(np.dot(
            eq_jphi_profile(geom, "jphi-linterp", eq=obj), J)) > 0 else -1.0
        out[tag] = {"pprime_sign": sgn,
                    "plasma_area": float(np.trapezoid(geom["dA_dpsiN"], psi_N)),
                    "vol_dpsi": float(np.trapezoid(
                        geom["dV_dpsi"], psi_N) * geom["dpsi_dpsiN"]),
                    "vol_dpsiN": float(np.trapezoid(geom["dV_dpsi"], psi_N))}
        for conv in ("jphi-linterp", "fsa"):
            j_eq = eq_jphi_profile(geom, conv, eq=obj, pprime_sign=sgn)
            out[tag][conv] = {
                "self": Ip_fsa_integral(obj, psi_N, j_eq, convention=conv,
                                        pprime_sign=sgn, geom=geom),
                "archived": Ip_fsa_integral(obj, psi_N, J, convention=conv,
                                            pprime_sign=sgn, geom=geom),
            }

    # the silent get_q collapse the clipping exists to avoid
    try:
        fsa_current_geometry(mygs, psi_N, psi_pad=0.0)
        out["collapse_guard"] = None
    except RuntimeError as exc:
        out["collapse_guard"] = str(exc)
    ravgs = mygs.get_q(psi=np.ascontiguousarray(psi_N))[2]
    R_raw = np.asarray(ravgs["<R>"] if isinstance(ravgs, dict) else ravgs[0],
                       dtype=float)
    out["unclipped_R_span"] = float(np.ptp(R_raw))

    out["Ip_after"] = float(mygs.compute_area_integral(mygs.calc_jtor_plasma()))
    with open(os.path.join(outdir, "fsa.json"), "w") as fh:
        json.dump(out, fh)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    work = tmp_path_factory.mktemp("fsa")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(work)],
        env=dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"FSA probe failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-4000:]}")
    with open(str(work / "fsa.json")) as fh:
        return json.load(fh)


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like mesh/baseline; skipped when unavailable")


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("convention", ["jphi-linterp", "fsa"])
def test_measure_reproduces_the_equilibriums_own_current(measured, convention):
    """THE validation: integrate the solved equilibrium's own j_phi profile and
    demand its true I_p back, to 0.1 %.  Measured +0.0071 %."""
    got = float(measured["live"][convention]["self"])
    err = abs(got / float(measured["Ip_true"]) - 1.0)
    assert err <= _SELF_CONSISTENCY, (
        f"[{convention}] Ip_fsa_integral of the equilibrium's own profile is "
        f"{got:.6e} against a true Ip of {measured['Ip_true']:.6e} "
        f"({100 * err:.4f}%, bar {100 * _SELF_CONSISTENCY:.1f}%)")


@pytest.mark.solver
@solver_only
def test_the_snapshot_geometry_equals_the_live_geometry(measured):
    """``_AnchorIpRenorm`` evaluates every getter on a ``copy_eq`` snapshot; if
    that were not bit-faithful the frozen-anchor discipline would be a lie."""
    for conv in ("jphi-linterp", "fsa"):
        for what in ("self", "archived"):
            assert measured["snapshot"][conv][what] == \
                measured["live"][conv][what], f"{conv}/{what}"
    assert measured["Ip_after"] == measured["Ip_true"], \
        "reading the snapshot's getters perturbed the live solver"


@pytest.mark.solver
@solver_only
def test_dV_dpsi_is_per_dimensional_psi(measured):
    """The Jacobian half of the formula.  The dpsi reading recovers the volume;
    the dpsi_N reading is out by ~300 %, which is how a missing Jacobian would
    present."""
    vol = float(measured["vol_true"])
    assert abs(measured["live"]["vol_dpsi"] / vol - 1.0) <= 0.01
    assert measured["live"]["vol_dpsiN"] / vol > 2.0


@pytest.mark.solver
@solver_only
def test_get_q_collapses_silently_on_an_unclipped_grid(measured):
    """Regression guard for the trap ``fsa_current_geometry`` clips around: an
    exact ``psi_N = 0`` sample makes every surface return the AXIS values, with
    no exception raised by OFT."""
    assert measured["unclipped_R_span"] < 1e-9, (
        "get_q no longer collapses on an unclipped grid -- if OFT fixed this, "
        "the clipping can stay but this test should be retired")
    assert measured["collapse_guard"], \
        "fsa_current_geometry accepted psi_pad=0 instead of raising"
    assert "collapsed" in measured["collapse_guard"]


@pytest.mark.solver
@solver_only
def test_compute_flux_integral_is_not_the_plasma_area(measured):
    """Documents defect 3 of ``_AnchorIpRenorm``: the mesh flux integral covers
    the limiter region, not the plasma, so it is not an I_p measure for a
    profile with a finite edge value."""
    fi_one = float(measured["flux_integral_of_one"])
    area = float(measured["live"]["plasma_area"])
    assert fi_one / area > 1.4, (
        f"compute_flux_integral(1) = {fi_one:.5f} m^2 is no longer much larger "
        f"than the plasma cross-section {area:.5f} m^2 -- if OFT changed the "
        f"interpolator's off-plasma behaviour, revisit the measure's rationale")


if __name__ == "__main__":
    _oft_importable()
    _probe(sys.argv[1])
